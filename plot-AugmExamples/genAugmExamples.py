import numpy as np
import cv2
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt

def removeBackground(image):
    if len(image.shape)==3:
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, 1, 2)
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w, :]
    elif len(image.shape)==2:
        _,mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, 1, 2)
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        return image[y:y+h, x:x+w]

def shearImage(image):
    shearV = 0.3
    shearMatrix_X = np.array([[1.0, shearV, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    shearV = 0.2
    shearMatrix_Y = np.array([[1.0, 0.0, 0.0],
                              [shearV, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    img = cv2.copyMakeBorder(image,225,225,225,225,cv2.BORDER_CONSTANT,value=[0,0,0])
    height, width, _ = img.shape
    shearAxis = np.random.choice([-1, 0, 1])
    img = cv2.warpPerspective(img, shearMatrix_X, (height,width))
    img = cv2.warpPerspective(img, shearMatrix_Y, (height,width))
    img = removeBackground(img)
    w, h, _ = img.shape
    img = cv2.copyMakeBorder(img, ((450-w)//2)+1, ((450-w)//2)+1,
                                  ((450-h)//2)+1, ((450-h)//2)+1,
                                  cv2.BORDER_CONSTANT,value=[0,0,0])
    img = img[0:450, 0:450, :]
    return img

def saltPepperNoise(image, salt_vs_pepper=0.2, amount=0.004):
    height, width, _ = image.shape
    #Create Salt and Pepper masks
    saltMask = np.random.uniform(0.0, 1.0, (height, width))
    pepperMask = np.random.uniform(0.0, 1.0, (height, width))
    saltMask[saltMask<=salt_vs_pepper] = 255.0
    pepperMask[pepperMask>salt_vs_pepper] = 255.0
    #pepperMask = pepperMask.astype(np.uint8)
    #saltMask = saltMask.astype(np.uint8)
    _, saltMask = cv2.threshold(saltMask, 127, 255, cv2.THRESH_BINARY)
    _, pepperMask = cv2.threshold(pepperMask, 127, 255, cv2.THRESH_BINARY)
    #Drop 'amount' pixels from saltMask
    toDrop = np.argwhere(saltMask>0)
    dropQtd = np.ceil(len(toDrop)*(1-amount)).astype(np.int32)
    np.random.shuffle(toDrop)
    for i in range(dropQtd):
        saltMask[toDrop[i][0],toDrop[i][1]] = 0
    #Drop 'amount' pixels from pepperMask
    toDrop = np.argwhere(pepperMask>0)
    dropQtd = np.ceil(len(toDrop)*(1-amount)).astype(np.int32)
    np.random.shuffle(toDrop)
    for i in range(dropQtd):
        pepperMask[toDrop[i][0],toDrop[i][1]] = 0
    #Apply pepperMasks on Image
    image = image.astype(np.int32)
    image[:,:,0] = np.add(image[:,:,0],pepperMask)
    image[:,:,1] = np.add(image[:,:,1],pepperMask)
    image[:,:,2] = np.add(image[:,:,2],pepperMask)
    #Apply saltMask on Image
    image[:,:,0] = np.subtract(image[:,:,0],saltMask)
    image[:,:,1] = np.subtract(image[:,:,1],saltMask)
    image[:,:,2] = np.subtract(image[:,:,2],saltMask)
    return np.clip(image, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    del matplotlib.font_manager.weight_dict['roman']
    matplotlib.font_manager._rebuild()

    #Fix Random Number Generation
    np.random.seed(12874)
    img = cv2.cvtColor(cv2.imread('UID_53_4_8_all.bmp'), cv2.COLOR_BGR2RGB)

    #************#
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    plt.imshow(img)
    plt.figtext(0.53, 0.015, '(a)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('augmEx-src-UID-53-4-8-all.png')
    plt.show()
    #************#
    #Mirroring
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    flippedImg = cv2.flip(img, -1)
    plt.imshow(flippedImg)
    plt.figtext(0.53, 0.015, '(b)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('augmEx-mirror-UID-53-4-8-all.png')
    plt.show()
    #************#
    #Rotation
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    theta = -60
    height, width, _ = img.shape
    rotationMatrix = cv2.getRotationMatrix2D((width/2,height/2), theta ,1)
    rotateImg = cv2.warpAffine(img, rotationMatrix,(width, height))
    plt.imshow(rotateImg)
    plt.figtext(0.53, 0.015, '(c)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('augmEx-rotate-UID-53-4-8-all.png')
    plt.show()
    #************#
    #Gaussian Blur
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    blurImg = cv2.GaussianBlur(img,(17,17),0)
    plt.imshow(blurImg)
    plt.figtext(0.53, 0.015, '(d)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('augmEx-gaussBlur-UID-53-4-8-all.png')
    plt.show()
    #************#
    #Shear
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    shearImg = shearImage(img)
    plt.imshow(shearImg)
    plt.figtext(0.53, 0.015, '(e)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('augmEx-shear-UID-53-4-8-all.png')
    plt.show()
    #************#
    #SaltnPepper
    plt.figure(figsize=(4.5, 5))
    rc('font', weight='bold')
    spImg = saltPepperNoise(img,salt_vs_pepper=0.50, amount = 0.05)
    plt.imshow(spImg)
    plt.figtext(0.53, 0.015, '(f)', wrap=True, horizontalalignment='center', fontname='Times New Roman', fontsize=24, fontweight='light')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('augmEx-saltpepper-UID-53-4-8-all.png')
    plt.show()
    #************#

    print(f"\nEnd Script!\n{'#'*50}")
