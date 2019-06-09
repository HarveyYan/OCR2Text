import cv2
import os
import shutil
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from skimage import data
from skimage.exposure import histogram
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import io

def ImgRead(path = '../Data/cell_images/training_set/', imgNum = 1):
    imgNum = len(os.listdir(path))
    imgs = []
    for i in range(imgNum):
        imgs.append(cv2.imread(path + 'clean' + str(i+1) + '.jpg', 0))
    
    return imgs


def EdgeDetect(img = cv2.imread('../Data/cell_images/training_set/1.jpg', 0), minVal = 100, maxVal = 200):
    edges = cv2.Canny(img, minVal, maxVal)

    return edges

def contourSeg(img):
    # ret, thresh = cv2.threshold(img, 191, 255, cv2.THRESH_TRUNC)
    # ret, thresh = cv2.threshold(img, 191, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

    delInd = []
    filteredContours = []
    for ind1, cnt1 in enumerate(contours):
        x1,y1,w1,h1 = cv2.boundingRect(cnt1)
        if w1 >= 0.8*img.shape[1] or w1 <= 1 or h1 <= 3:
            delInd.append(ind1)
            continue
        for ind2, cnt2 in enumerate(contours):
            if ind1 == ind2:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            if x1 <= x2 and y1 <= y2 and x1+w1 >= x2+w2 and y1+h1 >= y2+h2 :
                delInd.append(ind2)

    for ind, cnt in enumerate(contours):
        if ind not in delInd:
            filteredContours.append(cnt)

    return filteredContours

def chopImage(contours, imageToCut):
    cutImages = []
    x_axis = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        x_axis.append(x)
    ranked_x = np.argsort(x_axis)    

    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[ranked_x[i]])
        cutImages.append(imageToCut[y+2:y+h+3, x+2:x+w+3])
    return cutImages

def cutImagesWrite(path='../Data/cell_images/training_set/segments/', imageNum=1, cutImages=[]):
    dir = path + str(imageNum) + '/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    if len(cutImages) == 0:
        shutil.copyfile(path+'../BW/clean'+str(imageNum)+'.jpg', dir+str(imageNum)+'_'+str(0)+'.jpg')
    else:
        for ind, img in enumerate(cutImages):
            cv2.imwrite(dir + str(imageNum)+'_'+str(ind)+'.jpg', img)
    
    return True


# originalImgs = ImgRead('../Data/cell_images/training_set/')
BWimages = ImgRead('../Data/cell_images/training_set/BW/')


for i, BWI in enumerate(BWimages):
    BWI = BWimages[i]
    imageWidth = BWI.shape[0]
    imageHeight = BWI.shape[1]
    processedImg = BWI[2:imageWidth-3, 2:imageHeight-3]
    # processedImg = cv2.GaussianBlur(origin[3:imageWidth-3, 3:imageHeight-3], (3,3), 0)
    # edgedImg = EdgeDetect(processedImg, 50, 100)
    cts = contourSeg(processedImg)


    cutImgs = chopImage(cts, BWimages[i])
    cutImagesWrite('../Data/cell_images/training_set/segments/', i+1, cutImgs)


    # rectImg = BWimages[i]
    # for cnt in cts:
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     rectImg = cv2.rectangle(rectImg,(x+1,y+1),(x+w+4,y+h+4),(0,255,0),1)
    
    # x,y,w,h = cv2.boundingRect(cts[0])
    # cutimg = BWI[y+1:y+h+4, x+1:x+w+4]
    # plt.subplot(121),plt.imshow(BWI,cmap = 'gray')
    # plt.title('BW Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(cutimg, cmap = 'gray')
    # plt.title('rect Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
