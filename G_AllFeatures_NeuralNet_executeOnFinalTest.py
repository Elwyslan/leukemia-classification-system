from keras.layers import Dropout, Dense, LeakyReLU, Activation, PReLU, ELU
from keras import optimizers, regularizers
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard, Callback, LearningRateScheduler
from keras import backend as K
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import time
import copy
import os
import shutil
import numpy as np
import pandas as pd
import zipfile
from A_ExtractFeatures_Module import extractFeatureDict

imgsPath = Path('C-NMC_Leukemia/C-NMC_test_final_phase_data/C-NMC_test_final_phase_data/')
modelsPaths = Path('models/')

if __name__ == '__main__':
    #Read Train-Validation data
    print('Reading Train Dataframe...')
    train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1387-Features_40000-images.csv'), index_col=0)
    print('Done Read Train Dataframe!')

    columnsOrder = train_df.columns.tolist()
    columnsOrder.remove('cellType(ALL=1, HEM=-1)')

    imgs = {}
    for img in os.listdir(imgsPath):
        imgs[int(img.split('.')[0])] = imgsPath / img

    for modelFile in os.listdir(modelsPaths):
        modelPath = modelsPaths / modelFile
        model = load_model(str(modelPath))
        with open('isbi_valid.predict', 'w') as f:
            allCount=0
            hemCount=0
            for key in range(1,len(imgs)+1):
                featDict = extractFeatureDict(imgs[key])
                feats = []
                #Normalize and order features in feature vector
                for colName in columnsOrder:
                    feat = (featDict[colName] - train_df[colName].mean()) / train_df[colName].std() # mean=0 , std=1
                    feats.append(feat)

                predict = model.predict(np.array([feats],dtype=np.float32), verbose=1)
                if predict[0][0] > predict[0][1]:
                    f.write('1\n')
                    #print(f'{imgs[key]} - ALL')
                    allCount+=1
                else:
                    f.write('0\n')
                    #print(f'{imgs[key]} - HEM')
                    hemCount+=1
                print(f'Total Cells: 2586 (ALL: 1761, Normal: 825)\nTotal Cells: {allCount+hemCount:04d} (ALL: {allCount:04d}, Normal: {hemCount:03d})')
            f.flush()
        zipSubmission = zipfile.ZipFile(f'isbi_valid_{modelFile}.zip', 'w')
        zipSubmission.write('isbi_valid.predict', compress_type=zipfile.ZIP_DEFLATED)
        zipSubmission.close()
        os.remove('isbi_valid.predict')



    print(f"\nEnd Script!\n{'#'*50}")
