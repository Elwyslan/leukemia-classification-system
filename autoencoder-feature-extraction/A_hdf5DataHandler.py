import os
import cv2
import numpy as np
import h5py
from pathlib import Path
from collections import Counter

DATASET = Path('augmPatLvDiv')
TRAIN_DATA_PATH = Path('../data/augm_patLvDiv_train')

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
        print('Max:',np.max(img[:,:,0]), np.max(img[:,:,1]), np.max(img[:,:,2]))
        print('Min:',np.min(img[:,:,0]), np.min(img[:,:,1]), np.min(img[:,:,2]))
    #Process Grayscale Images
    elif colorScheme=='gray':
        #Read image
        img = cv2.cvtColor(cv2.imread(str(imgPath)), cv2.COLOR_BGR2GRAY)
        width, height = img.shape
        #Crop to required size
        w_crop = int(np.floor((width-imgSize[0])/2))
        h_crop = int(np.floor((height-imgSize[1])/2))
        img = img[w_crop:w_crop+imgSize[0], h_crop:h_crop+imgSize[1]]
        #Convert to Float 32
        img = np.array(img, dtype=np.float32)
        #Rescale
        img[:,:] = img[:,:]/255.0
        #Subtract mean
        if subtractMean:
            img[:,:] = img[:,:] - np.mean(img[:,:])
        #Divide by standard deviation
        if divideStdDev:
            img[:,:] = img[:,:] / np.std(img[:,:])
        #Expand dim to fit Tensorflow shape requirements
        img = np.expand_dims(img, axis=2)#Grayscale
    else:
        raise('Invalid color scheme')
    return img

def countCells(cellList):
    hemCount = allCount = 0
    for cell in cellList:
        if 'hem.bmp' in cell:
            hemCount+=1
        elif 'all.bmp' in cell:
            allCount+=1
    return hemCount, allCount


def createHDF5_train(noOfImages, subtractMean=True, divideStdDev=True, colorScheme='rgb', imgSize=(450,450)):
    #Read Images
    images = os.listdir(TRAIN_DATA_PATH)
    np.random.shuffle(images)

    """
    #Drop Augmented Images
    while True:
        dropIdx = None
        for idx in range(len(images)):
            if 'AugmentedImg' in images[idx]:
                dropIdx = idx
                break
        if isinstance(dropIdx, int):
            print(f'Dropped: {images[dropIdx]}')
            images.pop(dropIdx)
            continue
        else:
            break
    """

    #Drop images until desired size
    while len(images)>noOfImages:
        np.random.shuffle(images)
        hemCount, allCount = countCells(images)
        if hemCount==allCount:
            print(f'Dropped: {images.pop(0)}', end = '')
            print(f' - Remaing images: {len(images)}')
        else:
            for n, img in enumerate(images):
                if hemCount>allCount and 'hem.bmp' in img:
                    print(f'Dropped: {images.pop(n)}', end = '')
                    print(f' - Remaing images: {len(images)}')
                    break
                if allCount>hemCount and 'all.bmp' in img:
                    print(f'Dropped: {images.pop(n)}', end = '')
                    print(f' - Remaing images: {len(images)}')
                    break   
    
    #Tensorflow is channels-last
    if colorScheme=='gray':
        train_shape = (len(images), imgSize[0], imgSize[1], 1)
    elif colorScheme=='rgb':
        train_shape = (len(images), imgSize[0], imgSize[1], 3)
    else:
        raise('Invalid color scheme')
    
    #Create HDF5 matrices
    hdf5_file = h5py.File('HDF5data/Train_{}_{}imgs+{}_{}_{}.h5'.format(DATASET,
                                                                        len(images),
                                                                        imgSize[0],
                                                                        imgSize[1],
                                                                        train_shape[3]), mode='w')
    hdf5_file.create_dataset('train_imgs', train_shape, np.float32)

    #Store Images
    for n, image in enumerate(images):
        imgPath = '{}/{}'.format(TRAIN_DATA_PATH, image)
        hdf5_file['train_imgs'][n] = processImg(imgPath,
                                                subtractMean=subtractMean,
                                                divideStdDev=divideStdDev,
                                                colorScheme=colorScheme,
                                                imgSize=imgSize)
    #Close file
    hdf5_file.close()

if __name__ == '__main__':
    createHDF5_train(noOfImages=500, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(450,450))

    print("\nEnd Script!\n{}\n".format('#'*50))
