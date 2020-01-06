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
from sklearn.decomposition import PCA
import cv2
import multiprocessing
import sys
sys.path.append("..")
from A_ExtractFeatures_Module import extractFeatureDict

augm_patLvDiv_train = Path('../data/augm_patLvDiv_train/')
augm_patLvDiv_valid = Path('../data/augm_patLvDiv_valid')
patLvDiv_test = Path('../data/patLvDiv_test')


def testAutoEncoder(enc, dec, imgPath):
    src = cv2.cvtColor(cv2.imread(str(imgPath)), cv2.COLOR_BGR2RGB)
    pImg= processImg(imgPath, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(450,450))
    #Encode
    features = enc.predict(np.array([pImg]))
    print(f'Features: {features[0].tolist()}')
    #Decode
    predImg = dec.predict(features)[0]
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

def plotTSNE():
    np.random.seed(2987398)
    from sklearn.manifold import TSNE
    #df = Path('feature-dataframes/AugmPatLvDiv_TRAIN_250-Features_1001-images_91epochs.csv')
    #df = Path('feature-dataframes/AugmPatLvDiv_TRAIN_250-Features_1001-images_46epochs.csv')
    #df = Path('feature-dataframes/AugmPatLvDiv_TRAIN_250-Features_1001-images_146epochs.csv')
    df = Path('feature-dataframes/AugmPatLvDiv_TRAIN_250-Features_40000-images.csv')
    df = pd.read_csv(df, index_col=0)

    targets = df['cellType(ALL=1, HEM=-1)'].values
    df.drop(['cellType(ALL=1, HEM=-1)'], axis=1, inplace=True)

    #for col in df.columns:
    #    df[col] = (df[col] - df[col].mean()) / df[col].std() #mean=0, std=1
    
    tsne = TSNE(n_components=2, verbose=2, n_iter=4000, n_iter_without_progress=2000)

    X = tsne.fit_transform(df.values)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[targets==1,0], X[targets==1,1], marker='x', c='red', label='Células Malignas')
    plt.scatter(X[targets==-1,0], X[targets==-1,1], marker='x', c='blue', label='Células Saudáveis')
    plt.xlabel('X no espaço t-SNE',fontsize=18)
    plt.ylabel('Y no espaço t-SNE',fontsize=18)
    plt.legend(loc='upper right',prop={'size': 14, 'weight':'bold'})
    plt.tight_layout()
    plt.show()

    quit()

def computePCAMatrix(encoder, pcaMatrixPath):
    df = pd.DataFrame()
    listImgs = [augm_patLvDiv_train/imgFile for imgFile in os.listdir(augm_patLvDiv_train)]
    np.random.shuffle(listImgs)
    for n, imgPath in enumerate(listImgs):
        featuresDict = {}
        #Compute Cell name
        pImg= processImg(imgPath, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(450,450))
        feats = encoder.predict(np.array([pImg]))[0]
        for i in range(len(feats)):
            featuresDict[f'autoEncoder_feat{str(i).zfill(3)}'] = feats[i]
        print(f'Extracting PCA fit data - {str(n).zfill(5)} of {len(listImgs)} images')
        df = df.append(featuresDict, ignore_index=True)
    df.to_csv(Path(f'feature-dataframes/450x450-images_PCA-fitData_{df.shape[1]}-Features_{df.shape[0]}-images.csv'))
    print('\nFitting PCA.....\n')
    pca = PCA(n_components=125, svd_solver='full')
    pca.fit(df.values)
    print('\nFit PCA complete!\n')

    with open(Path(f'feature-dataframes/450x450-images_PCA-ReductionMatrix_{df.shape[1]}_to_{pca.components_.shape[0]}.pkl'), 'wb') as f:
        pickle.dump(pca, f, pickle.HIGHEST_PROTOCOL)

def extractAutoEncoderFeatures():
    #Load Encoder
    encoder = Path('Trained-AutoEncoder/84800942_Loss-12.898_Epoch-057-encoder.h5')
    encoder = load_model(str(encoder)) #Load encoder
    #Load PCA reduction Matrix
    pcaMatrix = Path('feature-dataframes/450x450-images_PCA-ReductionMatrix_250_to_125.pkl')
    if not pcaMatrix.is_file():
        computePCAMatrix(encoder=encoder, pcaMatrixPath=pcaMatrix)

    pca = None
    with open(pcaMatrix, 'rb') as f:
        pca = pickle.load(f)
    
    for mainDirectory in [(augm_patLvDiv_train, 'AugmPatLvDiv_TRAIN'), (augm_patLvDiv_valid, 'AugmPatLvDiv_VALIDATION'), (patLvDiv_test, 'PatLvDiv_TEST')]:
        #Create dataframe
        df = pd.DataFrame()
        #List images
        listImgs = [mainDirectory[0]/imgFile for imgFile in os.listdir(mainDirectory[0])]
        np.random.shuffle(listImgs)
        for n, imgPath in enumerate(listImgs):
            featuresDict = {}
            #Compute Cell type
            featuresDict['cellType(ALL=1, HEM=-1)'] = imgPath.name
            #Pre-process image
            pImg= processImg(imgPath, subtractMean=True, divideStdDev=False, colorScheme='rgb', imgSize=(450,450))
            #Encode Image
            feats = encoder.predict(np.array([pImg]))[0]
            #feats = pca.transform(feats.reshape(1, -1))[0]
            #Naming features
            for i in range(len(feats)):
                featuresDict[f'autoEncoder_PCA_feat{str(i).zfill(3)}'] = feats[i]
            #Append features to dataframe
            df = df.append(featuresDict, ignore_index=True)
            print(f'{mainDirectory[1]} - {df.shape[0]} images')
        #Save Dataframe
        cols = df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        df = df[cols]
        df.to_csv(f'feature-dataframes/450x450-images_{mainDirectory[1]}_{df.shape[1]-1}-Autoencoder-Features_{df.shape[0]}-images.csv')


def createDataframes():
    #Create TRAIN dataframe
    def createTrainDataframe():
        df = Path('feature-dataframes/450x450-images_AugmPatLvDiv_TRAIN_250-Autoencoder-Features_40000-images.csv')
        df = pd.read_csv(df, index_col=0)
        trainDF = pd.DataFrame()
        for n, row in enumerate(df.iterrows()):
            imgPath = augm_patLvDiv_train / str(row[1][0])
            feats = extractFeatureDict(imgPath)
            for i in range(row[1][1:].values.shape[0]):
                feats[f'autoEncoder_PCA_feat{str(i).zfill(3)}'] = row[1][1:].values[i]
            trainDF = trainDF.append(feats, ignore_index=True)
            print(f'Train - {n}')
        #Save Train Dataframe
        cols = trainDF.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        trainDF = trainDF[cols]
        trainDF.to_csv(f'feature-dataframes/450x450-images_AugmPatLvDiv_TRAIN_{trainDF.shape[1]-1}-Features_{trainDF.shape[0]}-images.csv')

    #Create VALIDATION dataframe
    def createValidDataframe():
        df = Path('feature-dataframes/450x450-images_AugmPatLvDiv_VALIDATION_250-Autoencoder-Features_10000-images.csv')
        df = pd.read_csv(df, index_col=0)
        validDF = pd.DataFrame()
        for n, row in enumerate(df.iterrows()):
            imgPath = augm_patLvDiv_valid / str(row[1][0])
            feats = extractFeatureDict(imgPath)
            for i in range(row[1][1:].values.shape[0]):
                feats[f'autoEncoder_PCA_feat{str(i).zfill(3)}'] = row[1][1:].values[i]
            validDF = validDF.append(feats, ignore_index=True)
            print(f'Valid - {n}')
        #Save Validation Dataframe
        cols = validDF.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        validDF = validDF[cols]
        validDF.to_csv(f'feature-dataframes/450x450-images_AugmPatLvDiv_VALIDATION_{validDF.shape[1]-1}-Features_{validDF.shape[0]}-images.csv')

    #Create TEST dataframe
    def createTestDataframe():
        df = Path('feature-dataframes/450x450-images_PatLvDiv_TEST_250-Autoencoder-Features_1503-images.csv')
        df = pd.read_csv(df, index_col=0)
        testDF = pd.DataFrame()
        for n, row in enumerate(df.iterrows()):
            imgPath = patLvDiv_test / str(row[1][0])
            feats = extractFeatureDict(imgPath)
            for i in range(row[1][1:].values.shape[0]):
                feats[f'autoEncoder_PCA_feat{str(i).zfill(3)}'] = row[1][1:].values[i]
            testDF = testDF.append(feats, ignore_index=True)
            print(f'Test - {n}')
        #Save Validation Dataframe
        cols = testDF.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        testDF = testDF[cols]
        testDF.to_csv(f'feature-dataframes/450x450-images_AugmPatLvDiv_TEST_{testDF.shape[1]-1}-Features_{testDF.shape[0]}-images.csv')
    

    p0 = multiprocessing.Process(name='train_AugmPatLvDiv', target=createTrainDataframe)
    p1 = multiprocessing.Process(name='valid_AugmPatLvDiv',target=createValidDataframe)
    p2 = multiprocessing.Process(name='test_PatLvDiv',target=createTestDataframe)
    p0.start()
    p1.start()
    p2.start()
    p0.join()
    p1.join()
    p2.join()



if __name__ == '__main__':
    #decoder = Path('Trained-AutoEncoder/84800942_Loss-12.898_Epoch-057-decoder.h5')
    #encoder = Path('Trained-AutoEncoder/84800942_Loss-12.898_Epoch-057-encoder.h5')
    #encoder = load_model(str(encoder)) #Load encoder
    #decoder = load_model(str(decoder)) #Load Decoder
    #print(encoder.summary())
    #listImgs = [augm_patLvDiv_train/imgFile for imgFile in os.listdir(augm_patLvDiv_train)]
    #testAutoEncoder(encoder, decoder, listImgs[5])
    #testAutoEncoder(encoder, decoder, listImgs[1])
    #testAutoEncoder(encoder, decoder, listImgs[0])
    #quit()
    #extractAutoEncoderFeatures()
    createDataframes()

    print(f"\nEnd Script!\n{'#'*50}")
    