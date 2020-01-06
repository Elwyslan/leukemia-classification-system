from keras.layers import Dropout, Dense, LeakyReLU, Activation, PReLU, ELU
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Reshape, Input, GlobalMaxPooling2D
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
from pathlib import Path

MAIN_LOSS_FUNCTION = 'mean_absolute_percentage_error' # mean_squared_error | mean_absolute_error | mean_absolute_percentage_error | MeanSquaredLogarithmicError
MAIN_METRIC = 'mae' # accuracy | mse | mae
MAIN_LR = 0.00001

class personalCallback(Callback):
    def __init__(self):
        self.lastEpochModel = []
        self.bestModels = []
        self.lowestLoss = 100000
        self.tStamp = str(time.time()).replace('.','')[-8:]
        return

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        preEpochModel = Path(f'Trained-AutoEncoder/{self.tStamp}_AutoEncoder_at_Epoch-{epoch:03d}.h5')
        self.model.save(str(preEpochModel))
        
        while len(self.lastEpochModel) > 0:
            os.remove(self.lastEpochModel.pop(0))

        self.lastEpochModel.append(preEpochModel)
        time.sleep(1)
        return
 
    def on_epoch_end(self, epoch, logs={}):
        loss = np.round(logs['loss'], decimals=3)
        saveModel = False
        if loss < self.lowestLoss:
            self.lowestLoss = loss
            saveModel = True

        if saveModel:
            #Split - ENCODER
            encoder = Model(inputs=self.model.get_layer('Encoder_1stConvBlock_0').input,
                            outputs=self.model.get_layer('Encoder_output').output)
            opt = optimizers.Adam(lr=MAIN_LR, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
            encoder.compile(loss=MAIN_LOSS_FUNCTION, 
                            optimizer=opt,
                            metrics=[MAIN_METRIC])

            #Split - DECODER
            decoderInput = Input((encoder.layers[-1].output_shape[1],)) #Decoder's input_shape matches with Encoder's output_shape
            decoder = None
            initDecodeLayer = False
            for n, ly in enumerate(self.model.layers):
                if initDecodeLayer:
                    decoder = self.model.layers[n] (decoder)
                    continue
                if 'Decoder' in ly.name:
                    decoder = self.model.layers[n] (decoderInput)
                    initDecodeLayer = True
            decoder = Model(inputs=decoderInput, outputs=decoder)
            opt = optimizers.Adam(lr=MAIN_LR, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
            decoder.compile(loss=MAIN_LOSS_FUNCTION,
                            optimizer=opt,
                            metrics=[MAIN_METRIC])

            autoencoderSavepath = Path(f'Trained-AutoEncoder/{self.tStamp}_Loss-{loss:.3f}_Epoch-{epoch+1:03d}-autoencoder.h5')
            encoderSavepath = Path(f'Trained-AutoEncoder/{self.tStamp}_Loss-{loss:.3f}_Epoch-{epoch+1:03d}-encoder.h5')
            decoderSavepath = Path(f'Trained-AutoEncoder/{self.tStamp}_Loss-{loss:.3f}_Epoch-{epoch+1:03d}-decoder.h5')

            self.model.save(str(autoencoderSavepath))
            encoder.save(str(encoderSavepath))
            decoder.save(str(decoderSavepath))

            while len(self.bestModels) > 0:
                os.remove(self.bestModels.pop(0))

            self.bestModels.append(autoencoderSavepath)
            self.bestModels.append(encoderSavepath)
            self.bestModels.append(decoderSavepath)
            time.sleep(1)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return


TRAIN_DATA_PATH = Path('HDF5data/Train_augmPatLvDiv_500imgs+250_250_3.h5')


if __name__ == '__main__':
    x_train = HDF5Matrix(TRAIN_DATA_PATH, 'train_imgs')
    #print(x_train.shape)

    kernelReg = regularizers.l1_l2(l1=0.0001, l2=0.0001)
    dropoutRate = 0.2

    #E N C O D E R
    autoEncoder_model = Sequential()
    
    #1st Conv. Block
    autoEncoder_model.add(Conv2D(input_shape=x_train.shape[1:], filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_1stConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_1stConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    
    #2nd Conv. Block
    autoEncoder_model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_2ndConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_2ndConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    
    #3rd Conv. Block
    autoEncoder_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_3rdConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_3rdConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

    #4th Conv. Block
    autoEncoder_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_4thConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_4thConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    
    #5th Conv. Block
    autoEncoder_model.add(Conv2D(filters=250, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_5thConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=250, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_5thConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(GlobalMaxPooling2D(name='Encoder_output'))
    
    #D E C O D E R 
    #250 = 5 x 5 x 5 x 2
    #(5x5) > (25x25) > x (75x75) > (225x225) > (450x450)
    #1st Conv. Block
    autoEncoder_model.add(Reshape((5,5,10), name='Decoder_reshape')) # Reshape (1, 250) -> (5,5,10)
    autoEncoder_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_1stConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_1stConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(UpSampling2D(size=(5, 5), interpolation='bilinear')) #interpolation = bilinear || nearest
    
    #2nd Conv. Block
    autoEncoder_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_2ndConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_2ndConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(UpSampling2D(size=(5, 5), interpolation='bilinear')) #interpolation = bilinear || nearest

    #3rd Conv. Block
    autoEncoder_model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_3rdConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_3rdConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(UpSampling2D(size=(2, 2), interpolation='bilinear')) #interpolation = bilinear || nearest

    #4th Conv. Layer
    autoEncoder_model.add(Conv2D(filters=32, kernel_size=(7,7), strides=(1,1), padding='same', name='Decoder_4thConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_4thConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    
    opt = optimizers.Adam(lr=MAIN_LR, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)

    #for ly in autoEncoder_model.layers[:-5]:
    #    ly.trainable = False

    autoEncoder_model.compile(loss=MAIN_LOSS_FUNCTION, # mean_squared_error | mean_absolute_error | mean_absolute_percentage_error | MeanSquaredLogarithmicError
                              optimizer=opt,
                              metrics=[MAIN_METRIC]) #'accuracy', 'mae', 'mse'
    
    print(autoEncoder_model.summary())

    #############################################
    #Training Parameters
    INIT_EPOCH = 0
    MAX_EPOCHS = 200
    BATCH_SIZE = 1
    LR_AT_EPOCH0 = MAIN_LR
    LR_AT_MAX_EPOCH = 0.000001

    #Learning rate decay - Update Learning rate
    def LR_decay(epoch):
        decayRate = (1/MAX_EPOCHS)*np.log(LR_AT_MAX_EPOCH/LR_AT_EPOCH0)
        return np.round(LR_AT_EPOCH0 * np.exp(decayRate*epoch), decimals=10)

    #Callbacks
    lrSched = LearningRateScheduler(LR_decay, verbose=1)
    pCallBack = personalCallback()
    logDir = Path(f'logdir/{pCallBack.tStamp}_autoencoder')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    tboard = TensorBoard(log_dir=logDir, write_graph=True)

    autoEncoder_model.fit(x_train, x_train,
                          initial_epoch=INIT_EPOCH,
                          epochs=MAX_EPOCHS,
                          batch_size=BATCH_SIZE,
                          shuffle='batch',
                          callbacks=[tboard, pCallBack, lrSched])

    print(f"\nEnd Script!\n{'#'*50}")
    #Train: https://drive.google.com/file/d/1XEFc5wfPFlqnQznhcYmFNg9wI5OMHNW2/view?usp=sharing
    #Valid: https://drive.google.com/file/d/1vF7GG2mz8dWJ-4zlKT0qS0sBbubWhQTb/view?usp=sharing
    #Test: https://drive.google.com/file/d/1fcK3_ZxDviN8v7ipknaugC-HdIZuhZEU/view?usp=sharing
