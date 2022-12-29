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
symbols = string.ascii_lowercase+ string.ascii_uppercase + "0123456789" 
# All symbols captcha can contain#-------------------------------------------------------
model=load_model('model_v4_5.h5')

def predict(filepath):
    
    '''output_folder = './output/'
    cv.imshow("321",img)
    file_name = '{}321.png'.format(output_folder)
    cv.imwrite(file_name, img)'''
    
    img = cv2.imread(filepath)
    cv2.imshow('original', img)
    cv2.waitKey()
    img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', img1)
    cv2.waitKey()
    
    img2=cv2.resize(img1,[300,100])
    cv2.imshow('300*100', img2)
    cv2.waitKey()
    kernel = np.ones((3,3), np.uint8)
    img3=cv2.dilate(img2,kernel,iterations = 1)
    cv2.imshow('dilate', img3)
    cv2.waitKey()
    ret,img4=cv2.threshold(img3,203,255,cv2.THRESH_BINARY)
    cv2.imshow('threshold', img4)
    cv2.waitKey()
    img5=cv2.resize(img4,[120,30])
    cv2.imshow('120*30', img5)
    cv2.waitKey()
    img=img5
    
    if img is not None:
        img = img / 255.0
    else:
        print("Not detected")
        return '100000'
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 62))#-------------------------------------------------------
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
        #probs.append(np.max(a))

    capt = ''
    for l in l_ind:
        capt += symbols[l]
    return capt#, sum(probs) / 5

print(predict('./test/TAe6i.png'))
