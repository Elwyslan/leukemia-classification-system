from keras.layers import Dropout, Dense, LeakyReLU, Activation, PReLU, ELU
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Reshape, Input
from keras import optimizers, regularizers
from keras.models import Sequential, Model, load_model
from keras.callbacks import TensorBoard, Callback, LearningRateScheduler
from keras import backend as K
from keras.utils.io_utils import HDF5Matrix
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import time
import copy
import os
import shutil
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
from A_hdf5DataHandler import processImg
import cv2

srcImgs = Path('../data/augm_patLvDiv_train/')
#decoder = Path('Trained-AutoEncoder/sampleDecoder5.h5')
#encoder = Path('Trained-AutoEncoder/sampleEncoder5.h5')
decoder = Path('Trained-AutoEncoder/15754793529632745-sampleDecoder_19.h5')
encoder = Path('Trained-AutoEncoder/15754793529632745-sampleEncoder_19.h5')

if __name__ == '__main__':
    
    encoder = load_model(str(encoder))
    decoder = load_model(str(decoder))

    listImgs = os.listdir(srcImgs)
    np.random.shuffle(listImgs)
    for imgFile in listImgs:
        src = cv2.cvtColor(cv2.imread(str(srcImgs/imgFile)), cv2.COLOR_BGR2RGB)
        pImg= processImg(srcImgs/imgFile, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(450,450))
        #Encode
        feats = encoder.predict(np.array([pImg]))
        print(feats[0])
        #Decode
        predImg = decoder.predict(feats)[0]

        plt.subplot(321)
        plt.hist(pImg[:,:,0])
        plt.subplot(322)
        plt.hist(predImg[:,:,0])

        plt.subplot(323)
        plt.hist(pImg[:,:,1])
        plt.subplot(324)
        plt.hist(predImg[:,:,1])

        plt.subplot(325)
        plt.hist(pImg[:,:,2])
        plt.subplot(326)
        plt.hist(predImg[:,:,2])

        plt.show()

        plt.subplot(121)
        plt.imshow(pImg)
        plt.subplot(122)
        plt.imshow(predImg)
        plt.show()
        
        quit()

    print(f"\nEnd Script!\n{'#'*50}")
    