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
    ###########################################################################################
    #                                                                                         #
    #   ####     #####     #####    #    #    #    #    ######    ######    #####      ####   # 
    #  #    #    #    #      #      #    ##  ##    #        #     #         #    #    #       # 
    #  #    #    #    #      #      #    # ## #    #       #      #####     #    #     ####   #
    #  #    #    #####       #      #    #    #    #      #       #         #####          #  #
    #  #    #    #           #      #    #    #    #     #        #         #   #     #    #  #
    #   ####     #           #      #    #    #    #    ######    ######    #    #     ####   #
    #                                                                                         #
    ###########################################################################################
    
    #Adadelta gridSearch
    rhos = [0.99]
    init_LRs = [0.95]
    finalLR = 0.4
    adaDeltaRuns = []
    kernelReg = regularizers.l1_l2(l1=0.001, l2=0.001)
    for rho in rhos:
        for initLR in init_LRs:
            runName = f'AdaDelta_initLR-{initLR:.4f}_finalLR-{finalLR:.2f}_rho-{rho:.2f}'
            #optimizers.Adadelta(lr=initLR, rho=rho)
            adaDeltaRuns.append(('AdaDelta', runName, initLR, rho, finalLR))

    optsList = adaDeltaRuns #adamRuns + adaDeltaRuns + sgdRuns

    ###########################################################################################################################################
    #                                                                                                                                         #
    #     #                                                #######                                                                            #
    #    # #       ####     #####    #    #    #           #          #    #    #    #     ####     #####    #     ####     #    #     ####   #
    #   #   #     #    #      #      #    #    #           #          #    #    ##   #    #    #      #      #    #    #    ##   #    #       #
    #  #     #    #           #      #    #    #           #####      #    #    # #  #    #           #      #    #    #    # #  #     ####   #
    #  #######    #           #      #    #    #           #          #    #    #  # #    #           #      #    #    #    #  # #         #  #
    #  #     #    #    #      #      #     #  #  ##        #          #    #    #   ##    #    #      #      #    #    #    #   ##    #    #  #
    #  #     #     ####       #      #      ##   ##        #           ####     #    #     ####       #      #     ####     #    #     ####   #
    #                                                                                                                                         #
    ###########################################################################################################################################

    actFuncList = ['relu']

    ###########################################################################################
    #                                                                                         #
    #  #    #    #    #####     #    #    #####          #####       ##      #####      ##    #
    #  #    ##   #    #    #    #    #      #            #    #     #  #       #       #  #   #
    #  #    # #  #    #    #    #    #      #            #    #    #    #      #      #    #  #
    #  #    #  # #    #####     #    #      #            #    #    ######      #      ######  #
    #  #    #   ##    #         #    #      #            #    #    #    #      #      #    #  #
    #  #    #    #    #          ####       #            #####     #    #      #      #    #  #
    #                                                                                         #
    ###########################################################################################

    
    #Read Train data

    print('Reading Train Dataframe...')
    train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1387-Features_40000-images.csv'), index_col=0)
    print('Done Read Train Dataframe!')

    print('Reading Validation Dataframe...')
    valid_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_VALIDATION-AllFeats_1387-Features_10000-images.csv'), index_col=0)
    print('Done Read Validation Dataframe!')

    print('Preparing Data...')

    x_train, y_train, x_valid, y_valid = prepareData(train_df=train_df, valid_df=valid_df)
    
    print('Done Read Train and Validation data!')

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
    

    ######################################################################################################
    #                                                                                                    #
    #   #####                                  #####                                                     #
    #  #     #    #####     #    #####        #     #    ######      ##      #####      ####     #    #  #
    #  #          #    #    #    #    #       #          #          #  #     #    #    #    #    #    #  #
    #  #  ####    #    #    #    #    #        #####     #####     #    #    #    #    #         ######  #
    #  #     #    #####     #    #    #             #    #         ######    #####     #         #    #  #
    #  #     #    #   #     #    #    #       #     #    #         #    #    #   #     #    #    #    #  #
    #   #####     #    #    #    #####         #####     ######    #    #    #    #     ####     #    #  #
    #                                                                                                    #
    ######################################################################################################

    def getActivationFunction(actFuncParams):
        if isinstance(actFuncParams, tuple):
            if actFuncParams[0] == 'LeakyReLU':
                return LeakyReLU(alpha=actFuncParams[1])
            elif actFuncParams[0] == 'ELU':
                return ELU(alpha=actFuncParams[1])
        elif isinstance(actFuncParams, str):
            if actFuncParams == 'PReLU':
                return PReLU()
            else:
                return Activation(actFuncParams)
        return None

    def getOptimizer(optParams):
        if optParams[0] == 'Adam':
            runName = optParams[1]
            initi_lr = optParams[2]
            final_lr = optParams[5]
            opt = optimizers.Adam(lr=initi_lr, decay=0.0 ,beta_1=optParams[3], beta_2=optParams[4], epsilon=1e-8, amsgrad=False)
        elif optParams[0] == 'AdaDelta':
            runName = optParams[1]
            initi_lr = optParams[2]
            final_lr = optParams[4]
            opt = optimizers.Adadelta(lr=initi_lr, rho=optParams[3])
        elif optParams[0] == 'SGD':
            runName = optParams[1]
            initi_lr = optParams[2]
            final_lr = optParams[4]
            opt = optimizers.SGD(lr=initLR, momentum=optParams[3], nesterov=False)

        return opt, runName, initi_lr, final_lr

    noOfNeurons = [128]
    noOfHiddenLayers = [2]
    for n_Neu in noOfNeurons:
        for n_Hidden in noOfHiddenLayers:
            for actFunc in actFuncList:
                for optChoice in optsList:
                    #Create Model
                    model = Sequential()
                    #Input Layer
                    model.add(Dense(n_Neu, input_shape=(x_train.shape[1],), kernel_regularizer=kernelReg))
                    model.add(getActivationFunction(actFunc))
                    model.add(Dropout(0.5))
                    #Hidden Layers
                    for _ in range(n_Hidden):
                        model.add(Dense(n_Neu, kernel_regularizer=kernelReg))
                        model.add(getActivationFunction(actFunc))
                        model.add(Dropout(0.5))
                    #Output Layer
                    model.add(Dense(2, activation='softmax', kernel_regularizer=kernelReg))
                    #Set Backpropagation Optimizer
                    opt, runName, initi_lr, final_lr = getOptimizer(optChoice)
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=opt,
                                  metrics=['accuracy'])

                    #TRAINING PARAMS
                    MAX_EPOCHS = 50
                    LR_AT_MAX_EPOCH = final_lr
                    LR_AT_EPOCH0 = initi_lr

                    #Log path
                    if isinstance(actFunc, tuple):
                        actName = f'{actFunc[0]}-{actFunc[1]}'
                    else:
                        actName = actFunc
                    logDir = Path(f'gridsearch-results/{actName}_HLs-{n_Hidden}_NEUs-{n_Neu}_' + runName)
                    logDir = Path(str(logDir) + '_L1L2-0.001_PCA95pct')
                    if not os.path.exists(Path(logDir)):
                        os.mkdir(Path(logDir))
                    else:
                        print(f'Skipped: {logDir}')
                        K.clear_session()
                        continue

                    #Learning rate decay - Update Learning rate
                    def LR_decay(epoch):
                        decayRate = (1/MAX_EPOCHS)*np.log(LR_AT_MAX_EPOCH/LR_AT_EPOCH0)
                        return np.round(LR_AT_EPOCH0 * np.exp(decayRate*epoch), decimals=10)

                    #Callbacks
                    lrSched = LearningRateScheduler(LR_decay, verbose=1)
                    tboard = TensorBoard(log_dir=logDir, write_graph=False)
                    
                    print(model.summary())
                    model.fit(x_train, y_train,
                              batch_size=1000,
                              epochs=MAX_EPOCHS,
                              validation_data=(x_valid, y_valid),
                              callbacks=[tboard, lrSched],
                              shuffle=True)
                    K.clear_session()

    print(f"\nEnd Script!\n{'#'*50}")

    #relu_HLs-2_NEUs-128_AdaDelta_initLR-1.0000_finalLR-0.10_rho-0.95