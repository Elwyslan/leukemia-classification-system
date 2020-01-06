from keras.layers import Input, Dropout, Dense, BatchNormalization,\
                         Activation, MaxPooling2D, Conv2D, Flatten,\
                         GlobalMaxPooling2D, GlobalAveragePooling2D,\
                         LeakyReLU, InputLayer, Lambda, concatenate
from keras import optimizers, regularizers, initializers
from keras.models import Sequential, Model, load_model
from keras.callbacks import TensorBoard, Callback, LearningRateScheduler
from keras.utils.io_utils import HDF5Matrix
from keras.utils.vis_utils import plot_model
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras import backend as K
from collections import Counter
import numpy as np
import os
import pandas as pd
from pathlib import Path
import time

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
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        modelSavepath = Path(f'models/BKP_ModifiedLeukonet_{self.tStamp}_Epoch-{epoch:03d}.h5')
        delModel = Path(f'models/BKP_ModifiedLeukonet_{self.tStamp}_Epoch-{epoch-1:03d}.h5')
        if delModel.is_file():
            os.remove(delModel)
        return
 
    def on_epoch_end(self, epoch, logs={}):
        acc = np.round(logs['val_acc'], decimals=4)
        saveModel = False
        if acc > self.highestAccuracy:
            self.highestAccuracy = acc
            saveModel = True

        if saveModel:
            modelSavepath = Path(f'models/ModifiedLeukonet_{self.tStamp}_Acc-{acc:.4f}_Epoch-{epoch+1:03d}.h5')

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


if __name__ == '__main__':

    ##################################################################
    #                                                                #
    #  #     #  #######  #######  #     #  #######  ######   #    #  #
    #  ##    #  #           #     #  #  #  #     #  #     #  #   #   #
    #  # #   #  #           #     #  #  #  #     #  #     #  #  #    #
    #  #  #  #  #####       #     #  #  #  #     #  ######   ###     #
    #  #   # #  #           #     #  #  #  #     #  #   #    #  #    #
    #  #    ##  #           #     #  #  #  #     #  #    #   #   #   #
    #  #     #  #######     #      ## ##   #######  #     #  #    #  #
    #                                                                #
    ##################################################################                                                        
    NETWORK_DROPOUT_RATE = 0.2
    KERNEL_REG = regularizers.l1_l2(l1=0.0001, l2=0.0001)

    #Input Layer
    inputLayer = Input(shape=(250,250,6))

    #RGB_ConvNet
    #The first three channels -> RGB image
    rgbConvNet = Lambda(lambda x : x[:,:,:,0:3]) (inputLayer)

    #1st Conv. Block
    rgbConvNet = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = Dropout(NETWORK_DROPOUT_RATE) (rgbConvNet)
    rgbConvNet = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (rgbConvNet)

    #1st Conv. Block
    rgbConvNet = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = Dropout(NETWORK_DROPOUT_RATE) (rgbConvNet)
    rgbConvNet = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (rgbConvNet)

    #2nd Conv. Block
    rgbConvNet = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = Dropout(NETWORK_DROPOUT_RATE) (rgbConvNet)
    rgbConvNet = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (rgbConvNet)

    #3rd Conv. Block
    rgbConvNet = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = Dropout(NETWORK_DROPOUT_RATE) (rgbConvNet)
    rgbConvNet = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (rgbConvNet)

    #4th Conv. Block
    rgbConvNet = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = Dropout(NETWORK_DROPOUT_RATE) (rgbConvNet)
    rgbConvNet = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (rgbConvNet)
    rgbConvNet = BatchNormalization() (rgbConvNet)
    rgbConvNet = Activation('relu') (rgbConvNet)
    rgbConvNet = GlobalMaxPooling2D() (rgbConvNet)

    #***********************************************************************************************#
    #***********************************************************************************************#

    #DCT_ConvNet
    #The last three channels -> DCT transform of each RGB channel
    dctConvNet = Lambda(lambda x : x[:,:,:,3:6]) (inputLayer)

    #1st Conv. Block
    dctConvNet = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = Dropout(NETWORK_DROPOUT_RATE) (dctConvNet)
    dctConvNet = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (dctConvNet)

    #1st Conv. Block
    dctConvNet = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = Dropout(NETWORK_DROPOUT_RATE) (dctConvNet)
    dctConvNet = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (dctConvNet)

    #2nd Conv. Block
    dctConvNet = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = Dropout(NETWORK_DROPOUT_RATE) (dctConvNet)
    dctConvNet = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (dctConvNet)

    #3rd Conv. Block
    dctConvNet = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = Dropout(NETWORK_DROPOUT_RATE) (dctConvNet)
    dctConvNet = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (dctConvNet)

    #4th Conv. Block
    dctConvNet = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = Dropout(NETWORK_DROPOUT_RATE) (dctConvNet)
    dctConvNet = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',kernel_regularizer=KERNEL_REG) (dctConvNet)
    dctConvNet = BatchNormalization() (dctConvNet)
    dctConvNet = Activation('relu') (dctConvNet)
    dctConvNet = GlobalMaxPooling2D() (dctConvNet)

    #***********************************************************************************************#
    #***********************************************************************************************#
    
    #Concatenate RGB ConvNet and DCT ConvNet
    merged = concatenate([rgbConvNet, dctConvNet], axis=1) # axis=1 >> Merge row, same column

    #Add Top Layer (Fully Connected)
    fullyConnected = Dense(512, activation='relu',kernel_regularizer=KERNEL_REG) (merged)
    fullyConnected = Dense(512, activation='relu',kernel_regularizer=KERNEL_REG) (fullyConnected)
    fullyConnected = Dense(2, activation='softmax',kernel_regularizer=KERNEL_REG) (fullyConnected)

    #Create and Compile
    model = Model(inputs=inputLayer, outputs=fullyConnected)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    

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
    TRAIN_DATA_PATH = Path('HDF5data/Train_AugmPatLvDiv_1000imgs+250_250_6.h5')
    VALIDATION_DATA_PATH = Path('HDF5data/Validation_AugmPatLvDiv_400imgs+250_250_6.h5')

    #Instantiating HDF5Matrix for the training set
    x_train = HDF5Matrix(TRAIN_DATA_PATH, 'train_imgs')
    y_train = HDF5Matrix(TRAIN_DATA_PATH, 'train_labels')

    #Instantiating HDF5Matrix for the validation set
    x_valid = HDF5Matrix(VALIDATION_DATA_PATH, 'valid_imgs')
    y_valid = HDF5Matrix(VALIDATION_DATA_PATH, 'valid_labels')

    
    #TRAINING PARAMS
    MAX_EPOCHS = 20
    LR_AT_EPOCH0 = 0.0001
    LR_AT_MAX_EPOCH = 0.00001
    opt = optimizers.Adam(lr=LR_AT_EPOCH0, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    #quit()
    #Learning rate decay - Update Learning rate
    def LR_decay(epoch):
        decayRate = (1/MAX_EPOCHS)*np.log(LR_AT_MAX_EPOCH/LR_AT_EPOCH0)
        return np.round(LR_AT_EPOCH0 * np.exp(decayRate*epoch), decimals=10)

    #Callbacks
    lrSched = LearningRateScheduler(LR_decay, verbose=1)
    pCallBack = personalCallback()
    logDir = Path(f'logdir/ModifiedLeukonet_{pCallBack.tStamp}')
    if not os.path.exists(Path(logDir)):
        os.mkdir(Path(logDir))
    tboard = TensorBoard(log_dir=logDir, write_graph=True)

    #Trainning
    model.fit(x_train, y_train,
              batch_size=1,
              epochs=MAX_EPOCHS,
              validation_data=(x_valid, y_valid),
              callbacks=[tboard, lrSched],
              shuffle='batch')
    K.clear_session()


    print("\nEnd Script!\n{}\n".format('#'*50))
