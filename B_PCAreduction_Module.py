import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import gc

if __name__ == '__main__':

    print('Reading Training Dataframe...')
    df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1612-Features_40000-images.csv'), index_col=0)
    print('Done Read Training Dataframe!')
    
    y = df['cellType(ALL=1, HEM=-1)']
    df.drop(['cellType(ALL=1, HEM=-1)'], axis=1, inplace=True)
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std() #mean=0, std=1
    X = df.values
    
    del df
    gc.collect()

    explainedVariances = [0.9, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99]

    for explainedVariance in explainedVariances:
        pca = PCA(n_components=explainedVariance, svd_solver='full')
        print(f'Fitting PCA reduction Matrix... (Explained Variance = {explainedVariance})')
        pca.fit(X)
        print('Fit Done!\n')
        savePath = Path(f'feature-dataframes/PCA_ReductionMatrix_ExplainedVariance-{explainedVariance}_{X.shape[1]}-TO-{pca.components_.shape[0]}.pkl')
        with open(savePath, 'wb') as f:
            pickle.dump(pca, f, pickle.HIGHEST_PROTOCOL)

    #pca = PCA(n_components=0.9, svd_solver='full') #763 components
    #pca = PCA(n_components=0.95, svd_solver='full') #940 components
    #pca = PCA(n_components=0.96, svd_solver='full') #981 components
    #pca = PCA(n_components=0.97, svd_solver='full') #1028 components
    #pca = PCA(n_components=0.98, svd_solver='full') #1080 components
    #pca = PCA(n_components=0.99, svd_solver='full') #1139 components

    print(f"\nEnd Script!\n{'#'*50}")

















    """

    from matplotlib import pyplot as plt
    imgPath = augm_patLvDiv_train / 'AugmentedImg_4055_UID_H2_6_2_hem.bmp'
    img = readImage(imgPath,color='gray')
    getContourSignature(img, sizeVector=320)
    print(f"\nEnd Script!\n{'#'*50}")



    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1001_UID_H23_10_1_hem.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1002_UID_H14_6_3_hem.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1003_UID_45_25_3_all.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1003_UID_H12_15_7_hem.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1003_UID_H22_17_7_hem.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1003_UID_H22_28_1_hem.bmp'), color='gray')
    img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1005_UID_H10_129_1_hem.bmp'), color='gray')

    from matplotlib import pyplot as plt
    #plt.imshow(img, cmap='gray')
    #plt.show()
    #getContourSignature(img, sizeVector=320)
    
    
    #Compute DCT transform
    img = np.array(img, dtype=np.float32)
    dctFeats = cv2.dct(img)
    #Thank's to https://stackoverflow.com/questions/50445847/how-to-zigzag-order-and-concatenate-the-value-every-line-using-python
    zigzagPattern = np.concatenate([np.diagonal(dctFeats[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctFeats.shape[0], dctFeats.shape[0])])
    zigzagPattern = zigzagPattern[0:1024]
    
    zigzagPattern = zigzagPattern.tolist()

    plt.subplot(311)
    plt.plot(zigzagPattern)
    plt.title('No_Log-Normalized dct', fontname='Times New Roman', fontsize=18)

    zigzagPattern = list(map(abs, zigzagPattern))

    plt.subplot(312)
    plt.plot(np.log(zigzagPattern))
    plt.title('Normal_Log-Normalized dct', fontname='Times New Roman', fontsize=18)

    plt.subplot(313)
    plt.plot(np.log10(zigzagPattern))
    plt.title('Log_10-Normalized dct', fontname='Times New Roman', fontsize=18)

    plt.show()
    """
