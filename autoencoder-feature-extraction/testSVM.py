from pathlib import Path
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm, neighbors, ensemble
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from matplotlib import pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

if __name__ == '__main__':
    #Read dataframes
    df = Path('feature-dataframes/450x450-images_PatLvDiv_TEST_250-Autoencoder-Features_1503-images.csv')
    df = pd.read_csv(df, index_col=0)
    """
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()
    quit()
    """
    #y = df['cellType(ALL=1, HEM=-1)'].values
    X = df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    X = X.values
    for i in range(X.shape[1]):
        print(X[:,i])
        if np.std(X[:,i])>-10e-8 and np.std(X[:,i])<10e-8:
            print('a')
        X[:,i] = (X[:,i] - np.mean(X[:,i])) / np.std(X[:,i])
        print(X[:,i])
        quit()
        #print(f'{np.mean(X[:,i])} - {np.std(X[:,i])}')
        #plt.hist(X[:,i], bins='auto')
        #plt.show()

    quit()

    #Pre-Process Data
    #pca = PCA(n_components=128)
    #pca.fit(X)
    #X = pca.transform(X)

    #with open(Path('feature-dataframes/450x450-images_PCA-ReductionMatrix_250_to_125.pkl'), 'wb') as f:
        #pickle.dump(pca, f, pickle.HIGHEST_PROTOCOL)
    #quit()

    #scaler = StandardScaler()
    #scaler.fit(X)
    #X = scaler.transform(X)

    #Linear Support Vector Classification
    rbf_svc = svm.SVC(C=1.0,
                      kernel='rbf',
                      gamma='auto',
                      shrinking=True,
                      tol=0.001,
                      cache_size=200,
                      verbose=True,
                      max_iter=-1)
    print('Fitting SVM.....\n')
    rbf_svc.fit(X, y)
    print('\nEnd Fit SVM.....')

    #Predict
    df = Path('feature-dataframes/AugmPatLvDiv_VALIDATION_250-Features_10000-images.csv')
    df = pd.read_csv(df, index_col=0)
    y_valid = df['cellType(ALL=1, HEM=-1)'].values
    X_valid = df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)

    y_pred = []
    y_true = list(y_valid)
    for i in range(len(X_valid)):
        pred = rbf_svc.predict(X_valid[i].reshape(1,-1))[0]
        if pred==1:
            y_pred.append('ALL')
        elif pred==-1:
            y_pred.append('HEM')

        if y_true[i]==1:
            y_true[i] = 'ALL'
        elif y_true[i]==-1:
            y_true[i] = 'HEM'
        print(f'{i} => True:{y_true[i]} - Predict:{y_pred[i]}')
    print(classification_report(y_true, y_pred))
    print(f'PCA n_components {pca.components_.shape[0]}')
    
    
    
    
    print(f"\nEnd Script!\n{'#'*50}")
