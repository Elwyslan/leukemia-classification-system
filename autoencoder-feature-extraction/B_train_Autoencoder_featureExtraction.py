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

class personalCallback(Callback):
    def __init__(self):
        self.bestModels = []
        self.lowestLoss = 100000
        self.tStamp = str(time.time()).replace('.','')[-8:]
        return

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
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
            opt = optimizers.Adam(lr=0.001, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
            encoder.compile(loss=MAIN_LOSS_FUNCTION, 
                            optimizer=opt,
                            metrics=['accuracy'])

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
            opt = optimizers.Adam(lr=0.001, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
            decoder.compile(loss=MAIN_LOSS_FUNCTION,
                            optimizer=opt,
                            metrics=['accuracy'])

            while len(self.bestModels) > 0:
                os.remove(self.bestModels.pop(0))
            autoencoderSavepath = Path(f'Trained-AutoEncoder/{self.tStamp}_Loss-{loss:.3f}_Epoch-{epoch+1:03d}-autoencoder.h5')
            encoderSavepath = Path(f'Trained-AutoEncoder/{self.tStamp}_Loss-{loss:.3f}_Epoch-{epoch+1:03d}-encoder.h5')
            decoderSavepath = Path(f'Trained-AutoEncoder/{self.tStamp}_Loss-{loss:.3f}_Epoch-{epoch+1:03d}-decoder.h5')

            self.model.save(str(autoencoderSavepath))
            encoder.save(str(encoderSavepath))
            decoder.save(str(decoderSavepath))

            self.bestModels.append(autoencoderSavepath)
            self.bestModels.append(encoderSavepath)
            self.bestModels.append(decoderSavepath)
            time.sleep(1)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return


TRAIN_DATA_PATH = Path('HDF5data/Train_augmPatLvDiv_500imgs+450_450_3.h5')


if __name__ == '__main__':
    x_train = HDF5Matrix(TRAIN_DATA_PATH, 'train_imgs', start=0, end=5)
    #print(x_train.shape)

    kernelReg = regularizers.l2(0.0001)
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
    autoEncoder_model.add(Conv2D(filters=250, kernel_size=(3,3), strides=(1,1), padding='same', name='Encoder_5thConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(Activation('relu'))
    autoEncoder_model.add(GlobalMaxPooling2D(name='Encoder_output'))
    
    #D E C O D E R 
    #450 = 5 x 5 x 3 x 3 x 2
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
    autoEncoder_model.add(UpSampling2D(size=(3, 3), interpolation='bilinear')) #interpolation = bilinear || nearest

    #3rd Conv. Block
    autoEncoder_model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_3rdConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_3rdConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(UpSampling2D(size=(3, 3), interpolation='bilinear')) #interpolation = bilinear || nearest

    #4th Conv. Layer
    autoEncoder_model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_4thConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_4thConvBlock_1', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(UpSampling2D(size=(2, 2), interpolation='bilinear')) #interpolation = bilinear || nearest
    

    #5th Conv. Layer
    autoEncoder_model.add(Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_5thConvBlock_0', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    autoEncoder_model.add(Dropout(dropoutRate))
    autoEncoder_model.add(Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', name='Decoder_output', kernel_regularizer=kernelReg))
    autoEncoder_model.add(PReLU(shared_axes=[1,2])) #autoEncoder_model.add(Activation('tanh'))
    
    opt = optimizers.Adam(lr=0.0001, decay=0.0 ,beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)

    for ly in autoEncoder_model.layers[:-5]:
        ly.trainable = False

    autoEncoder_model.compile(loss=MAIN_LOSS_FUNCTION, # mean_squared_error | mean_absolute_error | mean_absolute_percentage_error | MeanSquaredLogarithmicError
                              optimizer=opt,
                              metrics=['accuracy'])
    
    print(autoEncoder_model.summary())
    
    #############################################
    pCallBack = personalCallback()
    logDir = Path(f'logdir/{pCallBack.tStamp}_autoencoder')
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    tboard = TensorBoard(log_dir=logDir, write_graph=True)

    autoEncoder_model.fit(x_train, x_train,
                          epochs=100,
                          batch_size=1,
                          shuffle='batch',
                          callbacks=[tboard, pCallBack])

    print(f"\nEnd Script!\n{'#'*50}")
    
