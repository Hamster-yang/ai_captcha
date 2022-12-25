# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# %matplotlib inline 
import matplotlib.pyplot as plt
import os

from keras import layers
from keras.models import Model
from keras.models import load_model
from keras import callbacks
import os
import cv2
import string
import numpy as np
import random

#Init main values
'''symbols = string.ascii_lowercase+ string.ascii_uppercase + "0123456789" # All symbols captcha can contain#-------------------------------------------------------
num_symbols = len(symbols)
img_shape = (30,120 , 1)#-------------------------------------------------------
 
#print(num_symbols)



def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j,i] = 255
        elif img.ndim == 3:
            img[j,i,0]= 255
            img[j,i,1]= 255
            img[j,i,2]= 255
        return img

img = cv2.imread("C:/Users/allen/Desktop/ai_captcha/test/wMJ4l.png", cv2.IMREAD_GRAYSCALE)

result = salt(img, 500)
median = cv2.medianBlur(result, 5)
cv2.imshow("original_img", img)
cv2.imshow("Salt", result)
cv2.imshow("Median", median)
cv2.waitKey(0)
cv2.destroyWindow()'''

'''img = cv2.imread("C:/Users/allen/Desktop/ai_captcha/test/wMJ4l.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("1.Remove Noise", img)
img1=cv2.resize(img,[300,100])
#img1 = cv2.medianBlur(img, 3)
cv2.imshow("2.Remove Noise", img1)
kernel = np.ones((3,3), np.uint8)
img2=cv2.dilate(img1,kernel,iterations = 1)
cv2.imshow("3.Remove Noise", img2)
img3 = cv2.bitwise_or(img2,img1)
cv2.imshow("4.Remove Noise", img3)
ret,img4=cv2.threshold(img3,215,255,cv2.THRESH_BINARY)
cv2.imshow("5.Remove Noise", img4)
img5=cv2.erode(img4,kernel,iterations = 1)
cv2.imshow("6.Remove Noise", img5)

cv2.waitKey()'''

image = cv2.imread("/content/ai_captcha/test/QBsJp.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image,cmap='gray')
plt.show()
image=cv2.resize(image,[300,100])
kernel = np.ones((3,3), np.uint8)
image=cv2.dilate(image,kernel,iterations = 1)
ret,image=cv2.threshold(image,195,255,cv2.THRESH_BINARY)

image1=cv2.resize(image,[120,30])
plt.imshow(image1,cmap='gray')

plt.show()

'''img = cv2.imread("C:/Users/allen/Desktop/ai_captcha/test/wMJ4l.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("1.Remove Noise", img)
img1=cv2.resize(img,[300,100])
#img1 = cv2.medianBlur(img, 3)
cv2.imshow("2.Remove Noise", img1)
cv2.waitKey()
img2 = cv2.medianBlur(img1, 9)
cv2.imshow("3.Remove Noise", img2)
cv2.waitKey()
ret,img4=cv2.threshold(img2,140,255,cv2.THRESH_BINARY)
cv2.imshow("4.Remove Noise", img4)
cv2.waitKey()
img5 = cv2.bitwise_and(img4,img1)
cv2.imshow("5.Remove Noise", img5)
cv2.waitKey()
img3=cv2.resize(img5,[120,30])
#img1 = cv2.medianBlur(img, 3)
cv2.imshow("6.Remove Noise", img3)
cv2.waitKey()'''