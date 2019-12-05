import cv2
import numpy as np
import os
import shutil
from rawDataHandler import patients as HandlePatients
from pathlib import Path
import multiprocessing

"""
Remove cell's background
"""
def removeBackground(image):
    if len(image.shape)==3:
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, 1, 2)
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w, :]
    elif len(image.shape)==2:
        _,mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, 1, 2)
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w]

"""
Shear:
A shear parallel to the x axis results in:
    x' = x + shearV*y
    y' = y
A shear parallel to the y axis results in:
    x' = x
    y' = y + shearV*x
"""
def shearImage(image, s=(0.1, 0.35)):
    shearV = round(np.random.uniform(s[0], s[1]), 2)
    shearMatrix_X = np.array([[1.0, shearV, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    shearV = round(np.random.uniform(s[0], s[1]), 2)
    shearMatrix_Y = np.array([[1.0, 0.0, 0.0],
                              [shearV, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    img = cv2.copyMakeBorder(image,225,225,225,225,cv2.BORDER_CONSTANT,value=[0,0,0])
    height, width, _ = img.shape
    shearAxis = np.random.choice([-1, 0, 1])
    if shearAxis==-1:
        img = cv2.warpPerspective(img, shearMatrix_X, (height,width))
    elif shearV==1:
        img = cv2.warpPerspective(img, shearMatrix_Y, (height,width))
    else:
        img = cv2.warpPerspective(img, shearMatrix_X, (height,width))
        img = cv2.warpPerspective(img, shearMatrix_Y, (height,width))
    img = removeBackground(img)
    w, h, _ = img.shape
    img = cv2.copyMakeBorder(img, ((450-w)//2)+1, ((450-w)//2)+1,
                                  ((450-h)//2)+1, ((450-h)//2)+1,
                                  cv2.BORDER_CONSTANT,value=[0,0,0])
    img = img[0:450, 0:450, :]
    return img

"""
Salt and Pepper: Salt and Pepper noise refers to addition of white and black dots in the image. 
"""
def saltPepperNoise(image, salt_vs_pepper=0.2, amount=0.004):
    height, width, _ = image.shape
    #Create Salt and Pepper masks
    saltMask = np.random.uniform(0.0, 1.0, (height, width))
    pepperMask = np.random.uniform(0.0, 1.0, (height, width))
    saltMask[saltMask<=salt_vs_pepper] = 255.0
    pepperMask[pepperMask>salt_vs_pepper] = 255.0
    #pepperMask = pepperMask.astype(np.uint8)
    #saltMask = saltMask.astype(np.uint8)
    _, saltMask = cv2.threshold(saltMask, 127, 255, cv2.THRESH_BINARY)
    _, pepperMask = cv2.threshold(pepperMask, 127, 255, cv2.THRESH_BINARY)
    #Drop 'amount' pixels from saltMask
    toDrop = np.argwhere(saltMask>0)
    dropQtd = np.ceil(len(toDrop)*(1-amount)).astype(np.int32)
    np.random.shuffle(toDrop)
    for i in range(dropQtd):
        saltMask[toDrop[i][0],toDrop[i][1]] = 0
    #Drop 'amount' pixels from pepperMask
    toDrop = np.argwhere(pepperMask>0)
    dropQtd = np.ceil(len(toDrop)*(1-amount)).astype(np.int32)
    np.random.shuffle(toDrop)
    for i in range(dropQtd):
        pepperMask[toDrop[i][0],toDrop[i][1]] = 0
    #Apply pepperMasks on Image
    image = image.astype(np.int32)
    image[:,:,0] = np.add(image[:,:,0],pepperMask)
    image[:,:,1] = np.add(image[:,:,1],pepperMask)
    image[:,:,2] = np.add(image[:,:,2],pepperMask)
    #Apply saltMask on Image
    image[:,:,0] = np.subtract(image[:,:,0],saltMask)
    image[:,:,1] = np.subtract(image[:,:,1],saltMask)
    image[:,:,2] = np.subtract(image[:,:,2],saltMask)
    return np.clip(image, 0, 255).astype(np.uint8)


"""
Generate an Augmented Image base on srcImg
"""
def genAugmentedImage(srcImg):
    #Flip
    flipNum = int(np.random.choice([-1, 0, 1]))
    augm = cv2.flip(srcImg, flipNum)#horizontalANDVertical OR Horizontal OR vertical Flips
    #Rotation
    theta = np.random.randint(20, 340)
    height, width, _ = augm.shape
    rotationMatrix = cv2.getRotationMatrix2D((width/2,height/2), theta ,1)
    augm = cv2.warpAffine(augm, rotationMatrix,(width, height))
    #Aditional Augmentation
    rnd = np.random.uniform(0.0, 1.0)
    #25% Only Flip+Rotation
    if rnd < 0.25:
        return augm
    #25% Flip+Rotation+Shear
    if rnd >= 0.25 and rnd < 0.50:
        augm = shearImage(augm, s=(0.1, 0.35))
    #25% Flip+Rotation+PepperSalt
    if rnd >= 0.50 and rnd < 0.75:
        augm = saltPepperNoise(augm, salt_vs_pepper=0.50, amount = 0.05)
    #25% Flip+GaussianBlur
    if rnd >= 0.75:
        augm = cv2.GaussianBlur(augm,(5,5),0)
    return augm

def createDatasets(trainSize, validationSize):
    patLvDiv_train = Path('data/patLvDiv_train/')
    patLvDiv_valid = Path('data/patLvDiv_valid/')
    patLvDiv_test = Path('data/patLvDiv_test/')

    augm_patLvDiv_train = Path('data/augm_patLvDiv_train/')
    augm_patLvDiv_valid = Path('data/augm_patLvDiv_valid/')
    augm_rndDiv_train = Path('data/augm_rndDiv_train/')
    augm_rndDiv_valid = Path('data/augm_rndDiv_valid/')

    #Create folder 'data'
    if not os.path.exists(Path('data/')):
        os.mkdir(Path('data/'))
    else:
        shutil.rmtree(Path('data/'))
        os.mkdir(Path('data/'))

    #Create 5 Folders to store 5 datasets
    os.mkdir(patLvDiv_train)
    os.mkdir(patLvDiv_valid)
    os.mkdir(patLvDiv_test)
    os.mkdir(augm_patLvDiv_train)
    os.mkdir(augm_patLvDiv_valid)

    """
    #*************************************************************************#
    SPLIT DATASET BY PATIENT LEVEL - DIVISAO POR PACIENTE
    #*************************************************************************#
    """
    ALLtrainPat = []
    ALLvalidPat = []
    ALLtestPat = []
    
    #Retrieve patient IDs diagnosed with ALL
    ALLpatientsIDs = HandlePatients.getIdsALLPatients()

    #70% of ALL patientes for TRAIN
    maxPatTrain = int(np.ceil(len(ALLpatientsIDs)*0.7))

    #20% of ALL patientes for VALIDATION
    maxPatValid = int(np.floor(len(ALLpatientsIDs)*0.2))

    #Randomly populate 'ALLtrainPat' and 'ALLvalidPat' with TRAIN and VALIDATION patients
    while len(ALLpatientsIDs) > 0:
        np.random.shuffle(ALLpatientsIDs)
        if len(ALLtrainPat) < maxPatTrain:
            ALLtrainPat.append(ALLpatientsIDs.pop())
            continue
        if len(ALLvalidPat) < maxPatValid:
            ALLvalidPat.append(ALLpatientsIDs.pop())
            continue
        #Populate remaing patients for TEST
        ALLtestPat.append(ALLpatientsIDs.pop())

    HEMtrainPat = []
    HEMvalidPat = []
    HEMtestPat = []

    #Retrieve healthy patient IDs
    HEMpatientsIDs = HandlePatients.getIdsHEMPatients()

    #70% of ALL patientes for TRAIN
    maxPatTrain = int(np.ceil(len(HEMpatientsIDs)*0.7))

    #20% of ALL patientes for Validation
    maxPatValid = int(np.floor(len(HEMpatientsIDs)*0.2))
    
    #Randomly populate 'HEMtrainPat' and 'HEMvalidPat' with TRAIN and VALIDATION patients
    while len(HEMpatientsIDs) > 0:
        np.random.shuffle(HEMpatientsIDs)
        if len(HEMtrainPat) < maxPatTrain:
            HEMtrainPat.append(HEMpatientsIDs.pop())
            continue
        if len(HEMvalidPat) < maxPatValid:
            HEMvalidPat.append(HEMpatientsIDs.pop())
            continue
        #Populate remaing patients for TEST
        HEMtestPat.append(HEMpatientsIDs.pop())

    #Define TRAIN, VALIDATION and TEST patients IDs
    trainPat = HEMtrainPat+ALLtrainPat
    validPat = HEMvalidPat+ALLvalidPat
    testPat = HEMtestPat+ALLtestPat
    print(f'Patients ALL in Train: {len(ALLtrainPat)}, Patients HEM in Train: {len(HEMtrainPat)}')
    print(f'Patients ALL in Validation: {len(ALLvalidPat)}, Patients HEM in Validation: {len(HEMvalidPat)}')
    print(f'Patients ALL in Test: {len(ALLtestPat)}, Patients HEM in Test: {len(HEMtestPat)}')

    #For each Test patient...
    for pId in testPat:
        #Get patient cells
        pCells = HandlePatients.getPatientCellsPath(pId)
        #Copy each patient cells into folder 'patLvDiv_test'
        for cellpath in pCells:
            shutil.copy2(cellpath, patLvDiv_test)
            print(f'Copy {cellpath} TO {patLvDiv_test}/{cellpath.name}')

    #Prevent test dataset become too unbalanced
    count_testALL, count_testHEM = 0, 0
    for imgPath in os.listdir(patLvDiv_test):
        if 'all.bmp' in imgPath:
            count_testALL+=1
        elif 'hem.bmp' in imgPath:
            count_testHEM+=1

    ALL_HEM_ratio = 0.0
    if count_testALL > count_testHEM:
        ALL_HEM_ratio = count_testHEM/count_testALL
    elif count_testHEM > count_testALL:
        ALL_HEM_ratio = count_testALL/count_testHEM
    else:
        ALL_HEM_ratio = 0.5

    print(f'ALL-HEM ratio in Test dataset: {ALL_HEM_ratio}')
    #If TEST dataset is too unbalanced, repeat the process
    if ALL_HEM_ratio < 0.45:
        return createDatasets(trainSize, validationSize)

    #For each Train patient...
    for pId in trainPat:
        #Get patient cells
        pCells = HandlePatients.getPatientCellsPath(pId)
        #Copy each patient cells into folder 'patLvDiv_train'
        for cellpath in pCells:
            shutil.copy2(cellpath, patLvDiv_train)
            print(f'Copy {cellpath} TO {patLvDiv_train}/{cellpath.name}')

    #For each Validation patient...
    for pId in validPat:
        #Get patient cells
        pCells = HandlePatients.getPatientCellsPath(pId)
        #Copy each patient cells into folder 'patLvDiv_valid'
        for cellpath in pCells:
            shutil.copy2(cellpath, patLvDiv_valid)
            print(f'Copy {cellpath} TO {patLvDiv_valid}/{cellpath.name}')

    """
    #*************************************************************************#
    CREATE AUGMENTED IMAGES - DIVISAO POR PACIENTE
    #*************************************************************************#
    """
    #Thread to create 'Augm_patLvDiv_train'
    def createAugm_patLvDiv_train():
        countALL = 0 #Count how many ALL cells has in 'augm_patLvDiv_train'
        countHEM = 0 #Count how many HEM cells has in 'augm_patLvDiv_train'

        #For each cell in 'patLvDiv_train' folder
        for cellpath in os.listdir(patLvDiv_train):
            if 'all.bmp' in cellpath:
                countALL += 1
            elif 'hem.bmp' in cellpath:
                countHEM += 1
            else:
                continue
            #Copy cell into 'augm_patLvDiv_train' folder
            shutil.copy2(patLvDiv_train/cellpath, augm_patLvDiv_train)
            print(f'Copy {patLvDiv_train/cellpath} TO {augm_patLvDiv_train}/{cellpath}')

        #Read all cells in 'patLvDiv_train' folder
        srcTrain = os.listdir(patLvDiv_train)

        #Until 'augm_patLvDiv_train' folder didn't reach desired size...
        while len(os.listdir(augm_patLvDiv_train)) < trainSize:
            #Randomly choose a cell from 'patLvDiv_train' folder
            rndChoice = np.random.choice(srcTrain)
            #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
            if 'all.bmp' in rndChoice and countALL<trainSize//2:
                countALL += 1
            elif 'hem.bmp' in rndChoice and countHEM<trainSize//2:
                countHEM += 1
            else:
                print('\nERROR in augm_patLvDiv_train')
                print(f'Choice:{rndChoice}, countALL:{countALL}, countHEM:{countHEM}, trainSize//2:{trainSize//2}\n')
                continue
            #Create an augmented cell based on randomly choose from 'patLvDiv_train' folder
            img = patLvDiv_train / rndChoice
            try:
                img = genAugmentedImage(cv2.imread(str(img)))
            except Exception as e:
                print(str(e))
                #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
                if 'all.bmp' in rndChoice:
                    countALL -= 1
                elif 'hem.bmp' in rndChoice:
                    countHEM -= 1
                continue

            #Save the augmented image into folder 'augm_patLvDiv_train'
            savePath = augm_patLvDiv_train / f'AugmentedImg_{np.random.randint(1001, 9999)}_{rndChoice}'
            if not os.path.isfile(savePath):
                cv2.imwrite(str(savePath), img)
                print(f'Created {savePath}')
            else:
                #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
                if 'all.bmp' in rndChoice:
                    countALL -= 1
                elif 'hem.bmp' in rndChoice:
                    countHEM -= 1

    #Thread to create 'Augm_patLvDiv_valid'
    def createAugm_patLvDiv_valid():
        countALL = 0 #Count how many ALL cells has in 'augm_patLvDiv_train'
        countHEM = 0 #Count how many HEM cells has in 'augm_patLvDiv_train'

        #For each cell in 'patLvDiv_valid' folder
        for cellpath in os.listdir(patLvDiv_valid):
            if 'all.bmp' in cellpath:
                countALL += 1
            elif 'hem.bmp' in cellpath:
                countHEM += 1
            else:
                continue
            #Copy cell into 'augm_patLvDiv_valid' folder
            shutil.copy2(patLvDiv_valid/cellpath, augm_patLvDiv_valid)
            print(f'Copy {patLvDiv_valid/cellpath} TO {augm_patLvDiv_valid}/{cellpath}')

        #Read all cells in 'patLvDiv_valid' folder
        srcValid = os.listdir(patLvDiv_valid)

        #Until 'augm_patLvDiv_valid' folder didn't reach desired size...
        while len(os.listdir(augm_patLvDiv_valid)) < validationSize:
            #Randomly choose a cell from 'patLvDiv_valid' folder
            rndChoice = np.random.choice(srcValid)
            #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
            if 'all.bmp' in rndChoice and countALL<validationSize//2:
                countALL += 1
            elif 'hem.bmp' in rndChoice and countHEM<validationSize//2:
                countHEM += 1
            else:
                print('\nERROR in augm_patLvDiv_valid')
                print(f'Choice:{rndChoice}, countALL:{countALL}, countHEM:{countHEM}, validationSize//2:{validationSize//2}\n')
                continue
            #Create an augmented cell based on randomly choose from 'patLvDiv_valid' folder
            img = patLvDiv_valid / rndChoice
            try:
                img = genAugmentedImage(cv2.imread(str(img)))
            except Exception as e:
                print(str(e))
                #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
                if 'all.bmp' in rndChoice:
                    countALL -= 1
                elif 'hem.bmp' in rndChoice:
                    countHEM -= 1
                continue

            #Save the augmented image into folder 'augm_patLvDiv_valid'
            savePath = augm_patLvDiv_valid / f'AugmentedImg_{np.random.randint(1001, 9999)}_{rndChoice}'
            if not os.path.isfile(savePath):
                cv2.imwrite(str(savePath), img)
                print(f'Created {savePath}')
            else:
                #Logic to keep a balanced dataset (number of ALL cells equal to number of HEM cells)
                if 'all.bmp' in rndChoice:
                    countALL -= 1
                elif 'hem.bmp' in rndChoice:
                    countHEM -= 1

    #Create Augmented Datasets
    pTrain = multiprocessing.Process(name='Train Augm', target=createAugm_patLvDiv_train)
    pValid = multiprocessing.Process(name='Validation Augm', target=createAugm_patLvDiv_valid)
    pTrain.start()
    pValid.start()

    pTrain.join()
    pValid.join()


def folderData(folderPath):
    folderName = str(folderPath).split('/')[1]
    print(f'Folder Name: {folderName}')

    countALL = countHEM = countAugm_ALL_Imgs = countAugm_HEM_Imgs = 0
    srcALL_Imgs = srcHEM_Imgs = 0
    patientsIDs = []
    for cell in os.listdir(folderPath):
        if 'all.bmp' in cell:
            countALL += 1
            if 'AugmentedImg' in cell:
                countAugm_ALL_Imgs += 1
            else:
                srcALL_Imgs += 1
        elif 'hem.bmp' in cell:
            countHEM += 1
            if 'AugmentedImg' in cell:
                countAugm_HEM_Imgs += 1
            else:
                srcHEM_Imgs += 1

        if cell.split('_')[0] == 'AugmentedImg':
            patientID = cell.split('_')[3]
        else:
            patientID = cell.split('_')[1]
        patientsIDs.append(patientID)
    patientsIDs = list(set(patientsIDs))
    patientsIDs.sort()
    print(f'Source ALL cells: {srcALL_Imgs}')
    print(f'Source HEM cells: {srcHEM_Imgs}')
    print(f'Augmented ALL cells: {countAugm_ALL_Imgs}')
    print(f'Augmented HEM cells: {countAugm_HEM_Imgs}')
    print(f'Total of HEM cells: {countHEM}')
    print(f'Total of ALL cells: {countALL}')
    print(f'Dataset size: {countALL+countHEM} cells')
    HEM_IDs = HandlePatients.getIdsHEMPatients()
    ALL_IDs = HandlePatients.getIdsALLPatients()
    HEM_IDs = [pId for pId in HEM_IDs if pId in patientsIDs]
    ALL_IDs = [pId for pId in ALL_IDs if pId in patientsIDs]
    print(f'\nHealthy Patients IDs ({len(HEM_IDs)} patients): {HEM_IDs}\n')
    print(f'\nMalignant Patients IDs ({len(ALL_IDs)} patients): {ALL_IDs}\n')
    print(f'\nPatients IDs ({len(patientsIDs)} patients): {patientsIDs}\n')
    
    return countALL, countHEM, patientsIDs 


if __name__ == '__main__':
    #createDatasets(trainSize=40000, validationSize=10000)
    #folderData(Path('data/augm_patLvDiv_train/'))
    folderData(Path('data/augm_patLvDiv_valid/'))
    #folderData(Path('data/patLvDiv_test/'))
    #30000 = 25Gb
    #40000 = 31Gb
    """
    folderData(Path('data/augm_patLvDiv_train/'))
    Path('data/augm_patLvDiv_valid/')
    Path('data/augm_rndDiv_train/')
    Path('data/augm_rndDiv_valid/')
    Path('data/patLvDiv_test/')
    Path('data/patLvDiv_train/')
    Path('data/patLvDiv_valid/')
    _, _, pTrain = folderData(Path('data/rndDiv_train/'))
    _, _, pValid = folderData(Path('data/rndDiv_valid/'))
    _, _, pTest = folderData(Path('data/rndDiv_test/'))
    patients = pTrain + pValid + pTest
    patients = list(set(patients))
    patients.sort()
    
    countDict = {}
    for p in patients:
        check = ''
        if p in pTrain:
            check+='Train'
        if p in pValid:
            check+='Valid'
        if p in pTest:
            check+='Test'
        countDict[p] =check
    countList = list(countDict.values())
    from collections import Counter

    print(Counter(countList))

    print('*********************')
    print('*********************')
    _, _, _ = folderData(Path('data/patLvDiv_train/'))
    _, _, _ = folderData(Path('data/patLvDiv_valid/'))
    _, _, _ = folderData(Path('data/patLvDiv_test/'))
    """

    print(f"\nEnd Script!\n{'#'*50}")






















"""
    import matplotlib.pyplot as plt
    sample = Path('train/fold_1/all/UID_2_5_3_all.bmp')
    srcImg = cv2.imread(str(sample))
    srcImg = shearImage(srcImg, s=(0.2, 0.3))
    plt.imshow(srcImg)
    plt.show()
    quit()

    srcImg = srcImg[90:360,90:360,:]
    srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB)
    flipRot = cv2.flip(srcImg, 1)
    height, width, _ = flipRot.shape
    rotationMatrix = cv2.getRotationMatrix2D((width/2,height/2), 30 ,1)
    flipRot = cv2.warpAffine(flipRot, rotationMatrix,(width, height))
    shearImg, _ = shearImage(flipRot, s=(0.2, 0.25))
    spImg = saltPepperNoise(flipRot, salt_vs_pepper=0.50, amount = 0.01)


    
    fig=plt.figure()

    fig.add_subplot(1, 4, 1)
    plt.imshow(srcImg)
    plt.title('Source Image')
    fig.add_subplot(1, 4, 2)
    plt.imshow(flipRot)
    plt.title('Flip + Rotation')
    fig.add_subplot(1, 4, 3)
    plt.imshow(shearImg)
    plt.title('Flip + Rotation + Shear Transformation')
    fig.add_subplot(1, 4, 4)
    plt.imshow(spImg)
    plt.title("Flip + Rotation + Salt'nPepper noise")
    
    plt.show()


    05 dias: segunda 09/12/2019 à sexta 13/12/2019
    05 dias: segunda 16/12/2019 à sexta 20/12/2019

    24/12/2019 à 01/01/2020: Recesso natalino

    05 dias: segunda 06/01/2020 à sexta 10/01/2020
    05 dias: segunda 13/01/2020 à sexta 17/01/2019
    05 dias: segunda 20/01/2020 à sexta 24/01/2019
    05 dias: segunda 27/01/2020 à sexta 31/01/2020
    05 dias: segunda 03/02/2020 à sexta 07/02/2020

    35 dias

"""
