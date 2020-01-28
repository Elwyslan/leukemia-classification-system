import numpy as np
import cv2
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt    

def whiteBackGround(image):
    _,thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 1, 1, cv2.THRESH_BINARY)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if thresh[i,j]==0:
                image[i,j] = 255
    return image

if __name__ == '__main__':
    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()

    #Fix Random Number Generation
    np.random.seed(12874)
    img0 = cv2.cvtColor(cv2.imread('UID_1_2_2_all.bmp'), cv2.COLOR_BGR2RGB)
    #img0 = whiteBackGround(img0)
    img1 = cv2.cvtColor(cv2.imread('UID_1_3_1_all.bmp'), cv2.COLOR_BGR2RGB)
    #img1 = whiteBackGround(img1)
    img2 = cv2.cvtColor(cv2.imread('UID_h3_4_1_hem.bmp'), cv2.COLOR_BGR2RGB)
    #img2 = whiteBackGround(img2)
    img3 = cv2.cvtColor(cv2.imread('UID_h3_5_1_hem.bmp'), cv2.COLOR_BGR2RGB)
    #img3 = whiteBackGround(img3)

    #************#
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    plt.imshow(img0)
    plt.figtext(0.53, 0.015, '(a)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.tight_layout()
    plt.savefig('datasetSample-UID-1-2-2-all.png')
    plt.show()
    #************#
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    plt.imshow(img2)
    plt.figtext(0.53, 0.015, '(b)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.tight_layout()
    plt.savefig('datasetSample-UID-h3-4-1-hem.png')
    plt.show()
    #************#
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    plt.imshow(img1)
    plt.figtext(0.53, 0.015, '(c)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.tight_layout()
    plt.savefig('datasetSample-UID-1-3-1-all.png')
    plt.show()
    #************#
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    plt.imshow(img3)
    plt.figtext(0.53, 0.015, '(d)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.tight_layout()
    plt.savefig('datasetSample-UID-h3-5-1-hem.png')
    plt.show()
    #************#

    print(f"\nEnd Script!\n{'#'*50}")
