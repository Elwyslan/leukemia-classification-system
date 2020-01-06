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
    #Read Train-Validation data
    print('Reading Train Dataframe...')
    train_df = pd.read_csv(Path('feature-dataframes/450x450-images_AugmPatLvDiv_TRAIN_1637-Features_40000-images.csv'), index_col=0)
    print('Done Read Train Dataframe!')

    print('Reading Test Dataframe...')
    test_df = pd.read_csv(Path('feature-dataframes/450x450-images_AugmPatLvDiv_TEST_1637-Features_1503-images.csv'), index_col=0)
    print('Done Read Test Dataframe!')

    print('Preparing Data...')

    _, _, x_test, y_test = prepareData(train_df=train_df, valid_df=test_df)
    
    print('Done Read Train and Test data!')
    
    modelsPaths = Path('models/')
    result_df = pd.DataFrame(columns=['Model', 'ALL_Samples', 'HEM_Samples', 'True Positive', 'True Negative', 'False Positive', 'False Negative', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'Balanced Accuracy'])
    y_preds = []
    for modelFile in os.listdir(modelsPaths):
        modelPath = modelsPaths / modelFile
        model = load_model(str(modelPath))
        #Compute 'y_true' and 'y_pred'
        y_true = []
        y_pred = []
        for row_idx in range(x_test.shape[0]):
            #Ground Truth
            if np.array_equal(y_test[row_idx], ALL_LABEL):
                y_true.append('ALL')
            elif np.array_equal(y_test[row_idx], HEM_LABEL):
                y_true.append('HEM')
            else:
                continue
            
            #Predict
            pred = model.predict(np.array([x_test[row_idx, :]]), verbose=1)[0]
            
            #np.rint -> Round elements of the array to the nearest integer. [0.85 0.15] -> [1 0]
            if np.array_equal(np.rint(pred), ALL_LABEL):
                y_pred.append('ALL')
            elif np.array_equal(np.rint(pred), HEM_LABEL):
                y_pred.append('HEM')
            else:
                continue
            
        #Compute True Negative, False Positive, False Negative and True Positive
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=['HEM', 'ALL']).ravel()
        
        #Compute Metrics
        #Epsilon avoid division by zero
        epsilon = 1e-9
        #sensitivity/recall =  measures the proportion of actual positives that are correctly identified as such
        sensitivity = tp/(tp+fn+epsilon)
        #specificity = measures the proportion of actual negatives that are correctly identified as such
        specificity = tn/(tn+fp+epsilon)
        #accuracy = measures the systematic error
        accuracy = (tp+tn)/(tp+tn+fp+fn+epsilon)
        #precision = description of random errors, a measure of statistical variability.
        precision = tp/(tp+fp+epsilon)
        #The F1 score = can be interpreted as a average (harmonic mean) of the precision and sensitivity
        f1score = 2*((precision*sensitivity)/(precision+sensitivity))
        #The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
        #balancedAcc = balanced_accuracy_score(y_true, y_pred)
        balancedAcc = (sensitivity + specificity)/2

        sensitivity = np.round((sensitivity*100.0), decimals=2)
        specificity = np.round((specificity*100.0), decimals=2)
        accuracy = np.round((accuracy*100.0), decimals=2)
        precision = np.round((precision*100.0), decimals=2)
        f1score = np.round((f1score*100.0), decimals=2)
        balancedAcc = np.round((balancedAcc*100.0), decimals=2)

        print('###################')
        print(f'Sensitivity: {sensitivity}%')
        print(f'Specificity: {specificity}%')
        print(f'Accuracy: {accuracy}%')
        print(f'Precision: {precision}%')
        print(f'F1-score: {f1score}%')
        print(f'Balanced Accuracy: {balancedAcc}%')
        print('###################')

        #Store Results
        rowDict = {}
        rowDict['Model'] = modelFile
        rowDict['ALL_Samples'] = Counter(y_true)['ALL']
        rowDict['HEM_Samples'] = Counter(y_true)['HEM']
        rowDict['True Positive'] = tp
        rowDict['True Negative'] = tn
        rowDict['False Positive'] = fp
        rowDict['False Negative'] = fn
        rowDict['Accuracy'] = accuracy
        rowDict['Precision'] = precision
        rowDict['Sensitivity'] = sensitivity
        rowDict['Specificity'] = specificity
        rowDict['F1-Score'] = f1score
        rowDict['Balanced Accuracy'] = balancedAcc
        
        result_df = result_df.append(rowDict, ignore_index=True)

        y_preds.append(y_pred)

    
    result_df.to_csv(Path('results/NeuralNet_AllFeatures_TestPerformance.csv'))

    y_preds = np.array(y_preds)
    with open(Path('results/y_preds.txt'), 'w') as f:
        for idxImg in range(y_preds.shape[1]):
            print(f'{idxImg} - {Counter(y_preds[:,idxImg])}')
            f.write(f'{idxImg} - {Counter(y_preds[:,idxImg])}\n')

    print(f"\nEnd Script!\n{'#'*50}")
