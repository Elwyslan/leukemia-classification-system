from keras.layers import Dropout, Dense, LeakyReLU, Activation, PReLU, ELU
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.callbacks import TensorBoard, Callback, LearningRateScheduler
from keras import backend as K
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


ALL_LABEL = [1.0, 0.0]
HEM_LABEL = [0.0, 1.0]


class personalCallback(Callback):
    def __init__(self):
        self.bestModels = []
        self.highestAccuracy = -1
        self.tStamp = str(time.time()).replace('.','')[-8:]
        return

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        #acc = np.round(logs['val_acc'], decimals=4)
        #modelSavepath = Path(f'models/{self.tStamp}_Acc-{acc:.4f}_FinalEpoch-NeuralNet.h5')
        #self.model.save(str(modelSavepath))
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        acc = np.round(logs['val_acc'], decimals=4)
        saveModel = False
        if acc > self.highestAccuracy:
            self.highestAccuracy = acc
            saveModel = True

        if saveModel:
            modelSavepath = Path(f'models/{self.tStamp}_Acc-{acc:.4f}_Epoch-{epoch+1:03d}-NeuralNet_PCA95pct.h5')

            self.model.save(str(modelSavepath))

            while len(self.bestModels) > 0:
                os.remove(self.bestModels.pop(0))

            self.bestModels.append(modelSavepath)

            time.sleep(1)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return


def prepareData(train_df, valid_df):
    #Prepare Validation data
    y_valid = list(valid_df['cellType(ALL=1, HEM=-1)'].values)
    for i in range(len(y_valid)):
        if y_valid[i]==-1:
            y_valid[i] = HEM_LABEL
        elif y_valid[i]==1:
            y_valid[i] = ALL_LABEL
    y_valid = np.array(y_valid)
    x_valid = valid_df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    for col in x_valid.columns:
        x_valid[col] = (x_valid[col] - train_df[col].mean()) / train_df[col].std() #mean=0, std=1
    x_valid = x_valid.values

    #Prepare Train data
    y_train = list(train_df['cellType(ALL=1, HEM=-1)'].values)
    for i in range(len(y_train)):
        if y_train[i]==-1:
            y_train[i] = HEM_LABEL
        elif y_train[i]==1:
            y_train[i] = ALL_LABEL
    y_train = np.array(y_train)
    x_train = train_df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    for col in x_train.columns:
        x_train[col] = (x_train[col] - train_df[col].mean()) / train_df[col].std() #mean=0, std=1
    x_train = x_train.values
    return x_train, y_train, x_valid, y_valid

if __name__ == '__main__':
    #Read Train-Validation data
    print('Reading Train Dataframe...')
    train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1387-Features_40000-images.csv'), index_col=0)
    print('Done Read Train Dataframe!')

    print('Reading Validation Dataframe...')
    valid_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_VALIDATION-AllFeats_1387-Features_10000-images.csv'), index_col=0)
    print('Done Read Validation Dataframe!')

    print('Preparing Data...')

    x_train, y_train, x_valid, y_valid = prepareData(train_df=train_df, valid_df=valid_df)

    #Load PCA reduction Matrix
    pca = None
    with open(Path('feature-dataframes/PCA_ReductionMatrix_ExplainedVariance-0.95_1387-TO-951.pkl'), 'rb') as f:
        pca = pickle.load(f)
    
    print('PCA reducing input data dimensionality...')
    x_train = pca.transform(x_train)
    x_valid = pca.transform(x_valid)
    print('Input data dimensionality reduced!')

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    
    print('Done Read Train and Validation data!')

    kernelReg = regularizers.l1_l2(l1=0.001, l2=0.001)
    model = Sequential()
    model.add(Dense(128, input_shape=(x_train.shape[1],), kernel_regularizer=kernelReg))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_regularizer=kernelReg))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_regularizer=kernelReg))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', kernel_regularizer=kernelReg))

    #TRAINING PARAMS
    MAX_EPOCHS = 500
    LR_AT_EPOCH0 = 1.0
    LR_AT_MAX_EPOCH = 0.4
    
    opt = optimizers.Adadelta(lr=LR_AT_EPOCH0, rho=0.99)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    #relu_HLs-2_NEUs-128_AdaDelta_initLR-1.0000_finalLR-0.40_rho-0.99_L1L2-0.001_PCA95pct
    print(model.summary())
    
    #Learning rate decay - Update Learning rate
    def LR_decay(epoch):
        decayRate = (1/MAX_EPOCHS)*np.log(LR_AT_MAX_EPOCH/LR_AT_EPOCH0)
        return np.round(LR_AT_EPOCH0 * np.exp(decayRate*epoch), decimals=10)

    #Callbacks
    lrSched = LearningRateScheduler(LR_decay, verbose=1)
    pCallBack = personalCallback()

    logDir = Path(f'logdir/{pCallBack.tStamp}_relu_HLs-2_NEUs-128_AdaDelta_initLR-1.0000_finalLR-0.40_rho-0.99_L1L2-0.001_PCA95pct')
    if not os.path.exists(Path(logDir)):
        os.mkdir(Path(logDir))
    tboard = TensorBoard(log_dir=logDir, write_graph=False)
    
    
    model.fit(x_train, y_train,
              batch_size=1000,
              epochs=MAX_EPOCHS,
              validation_data=(x_valid, y_valid),
              callbacks=[tboard, lrSched, pCallBack],
              shuffle=True)
    K.clear_session()

    print(f"\nEnd Script!\n{'#'*50}")
