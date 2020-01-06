import os
import cv2
import numpy as np
import h5py
from pathlib import Path

TRAIN_DATA_PATH = Path('../data/augm_patLvDiv_train')
VALIDATION_DATA_PATH = Path('../data/augm_patLvDiv_valid')

ALL_LABEL = [[1.0, 0.0]]
HEM_LABEL = [[0.0, 1.0]]

def processImg(imgPath, subtractMean, divideStdDev, colorScheme, imgSize):
    #Process RGB Images
    if colorScheme=='rgb':
        #Read Image
        img = cv2.cvtColor(cv2.imread(str(imgPath)), cv2.COLOR_BGR2RGB)
        width, height, _ = img.shape
        #Crop to required size
        w_crop = int(np.floor((width-imgSize[0])/2))
        h_crop = int(np.floor((height-imgSize[1])/2))
        img = img[w_crop:w_crop+imgSize[0], h_crop:h_crop+imgSize[1], :]
        #Convert to Float 32
        img = np.array(img, dtype=np.float32)
        #Compute DCT for R-G-B channels
        dctFeats_r = cv2.dct(img[:,:,0]) + 10e-8 #add a small constant in order to avoid NaN in log10
        dctFeats_g = cv2.dct(img[:,:,1]) + 10e-8 #add a small constant in order to avoid NaN in log10
        dctFeats_b = cv2.dct(img[:,:,2]) + 10e-8 #add a small constant in order to avoid NaN in log10
        #DCT - R
        signal = np.array(dctFeats_r)
        signal[signal<0] = -1
        signal[signal>0] = 1
        dctFeats_r = np.log10(np.abs(dctFeats_r))
        dctFeats_r = np.multiply(signal,dctFeats_r)
        #DCT - G
        signal = np.array(dctFeats_g)
        signal[signal<0] = -1
        signal[signal>0] = 1
        dctFeats_g = np.log10(np.abs(dctFeats_g))
        dctFeats_g = np.multiply(signal,dctFeats_g)
        #DCT - B
        signal = np.array(dctFeats_b)
        signal[signal<0] = -1
        signal[signal>0] = 1
        dctFeats_b = np.log10(np.abs(dctFeats_b))
        dctFeats_b = np.multiply(signal,dctFeats_b)

        #dctFeats_r = (dctFeats_r - np.mean(dctFeats_r))/np.std(dctFeats_r)
        dctFeats_r = dctFeats_r - np.mean(dctFeats_r)
        #dctFeats_g = (dctFeats_g - np.mean(dctFeats_g))/np.std(dctFeats_g)
        dctFeats_g = dctFeats_g - np.mean(dctFeats_g)
        #dctFeats_b = (dctFeats_b - np.mean(dctFeats_b))/np.std(dctFeats_b)
        dctFeats_b = dctFeats_b - np.mean(dctFeats_b)

        #****************************************************#
        
        #Rescale
        img[:,:,0] = img[:,:,0]/255.0
        img[:,:,1] = img[:,:,1]/255.0
        img[:,:,2] = img[:,:,2]/255.0
        #Subtract mean
        if subtractMean:
            #print('Mean:',np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))
            img[:,:,0] = img[:,:,0] - np.mean(img[:,:,0])
            img[:,:,1] = img[:,:,1] - np.mean(img[:,:,1])
            img[:,:,2] = img[:,:,2] - np.mean(img[:,:,2])
        #Divide standard deviation
        if divideStdDev:
            #print('std:',np.std(img[:,:,0]), np.std(img[:,:,1]), np.std(img[:,:,2]))
            img[:,:,0] = img[:,:,0] / np.std(img[:,:,0])
            img[:,:,1] = img[:,:,1] / np.std(img[:,:,1])
            img[:,:,2] = img[:,:,2] / np.std(img[:,:,2])
        #print('Max:',np.max(img[:,:,0]), np.max(img[:,:,1]), np.max(img[:,:,2]))
        #print('Min:',np.min(img[:,:,0]), np.min(img[:,:,1]), np.min(img[:,:,2]))
    #Process Grayscale Images
    elif colorScheme=='gray':
        raise NotImplementedError('Gray images - To be implemented')
    else:
        raise('Invalid color scheme')
    #Stack RGB+DCT (6 channels)
    return np.dstack((img, np.dstack((dctFeats_r, dctFeats_g, dctFeats_b))))

def createHDF5_train(noOfImages, subtractMean=True, divideStdDev=True, colorScheme='rgb', imgSize=(450,450)):
    #Read Images
    images = os.listdir(TRAIN_DATA_PATH)
    np.random.shuffle(images)
    
    #Drop images until desired size
    droppedALL = droppedHEM = 0
    dropQtd = np.ceil((len(images) - noOfImages)/2)
    while len(images)>noOfImages:
        if ('hem.bmp' in images[0]) and (droppedHEM<dropQtd):
            droppedHEM+=1
            images.pop(0)
        if ('all.bmp' in images[0]) and (droppedALL<dropQtd):
            droppedALL+=1
            images.pop(0)
        np.random.shuffle(images)
        print(f'ALL: {droppedALL:05d}, HEM: {droppedHEM:05d}')

    #Tensorflow is channels-last
    if colorScheme=='gray':
        raise NotImplementedError('Gray images - To be implemented')
    elif colorScheme=='rgb':
        train_shape = (len(images), imgSize[0], imgSize[1], 6)
    else:
        raise('Invalid color scheme')
    
    #Create HDF5 matrices
    hdf5_file = h5py.File(Path(f'HDF5data/Train_AugmPatLvDiv_{train_shape[0]}imgs+{train_shape[1]}_{train_shape[2]}_{train_shape[3]}.h5'), mode='w')
    hdf5_file.create_dataset('train_imgs', train_shape, np.float32)
    hdf5_file.create_dataset('train_labels', (len(images),2), np.float32)
    
    images = [TRAIN_DATA_PATH/img for img in images]

    #Store Images
    for n, imgPath in enumerate(images):
        if 'all.bmp' in imgPath.name:
            print(n,'train-ALL')
            hdf5_file['train_labels'][n] = ALL_LABEL
        if 'hem.bmp' in imgPath.name:
            print(n,'train-HEM')    
            hdf5_file['train_labels'][n] = HEM_LABEL
        hdf5_file['train_imgs'][n] = processImg(imgPath,
                                                subtractMean=subtractMean,
                                                divideStdDev=divideStdDev,
                                                colorScheme=colorScheme,
                                                imgSize=imgSize)
    #Close file
    hdf5_file.close()

def createHDF5_validation(noOfImages, subtractMean=True, divideStdDev=True, colorScheme='rgb', imgSize=(450,450)):
    #Read Images
    images = os.listdir(VALIDATION_DATA_PATH)
    np.random.shuffle(images)
    
    #Drop images until desired size
    droppedALL = droppedHEM = 0
    dropQtd = np.ceil((len(images) - noOfImages)/2)
    while len(images)>noOfImages:
        if ('hem.bmp' in images[0]) and (droppedHEM<dropQtd):
            droppedHEM+=1
            images.pop(0)
        if ('all.bmp' in images[0]) and (droppedALL<dropQtd):
            droppedALL+=1
            images.pop(0)
        np.random.shuffle(images)
        print(f'ALL: {droppedALL:05d}, HEM: {droppedHEM:05d}')

    #Tensorflow is channels-last
    if colorScheme=='gray':
        raise NotImplementedError('Gray images - To be implemented')
    elif colorScheme=='rgb':
        valid_shape = (len(images), imgSize[0], imgSize[1], 6)
    else:
        raise('Invalid color scheme')
    
    #Create HDF5 matrices
    hdf5_file = h5py.File(Path(f'HDF5data/Validation_AugmPatLvDiv_{valid_shape[0]}imgs+{valid_shape[1]}_{valid_shape[2]}_{valid_shape[3]}.h5'), mode='w')
    hdf5_file.create_dataset('valid_imgs', valid_shape, np.float32)
    hdf5_file.create_dataset('valid_labels', (len(images),2), np.float32)
    
    images = [VALIDATION_DATA_PATH/img for img in images]

    #Store Images
    for n, imgPath in enumerate(images):
        if 'all.bmp' in imgPath.name:
            print(n,'valid-ALL')
            hdf5_file['valid_labels'][n] = ALL_LABEL
        if 'hem.bmp' in imgPath.name:
            print(n,'valid-HEM')    
            hdf5_file['valid_labels'][n] = HEM_LABEL
        hdf5_file['valid_imgs'][n] = processImg(imgPath,
                                                subtractMean=subtractMean,
                                                divideStdDev=divideStdDev,
                                                colorScheme=colorScheme,
                                                imgSize=imgSize)
    #Close file
    hdf5_file.close()


if __name__ == '__main__':
    import multiprocessing
    def createValidationSet():
        createHDF5_validation(noOfImages=400, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(250,250))
    
    def createTrainSet():
        createHDF5_train(noOfImages=1000, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(250,250))
    #Spawn Process
    pTrain = multiprocessing.Process(name='HDF5_Train', target=createTrainSet)
    pValid = multiprocessing.Process(name='HDF5_Validation',target=createValidationSet)
    pTrain.start()
    pValid.start()
    pTrain.join()
    pValid.join()

    print("\nEnd Script!\n{}\n".format('#'*50))
