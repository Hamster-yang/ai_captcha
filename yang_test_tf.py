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
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    img=cv2.resize(img,[300,100])
    kernel = np.ones((3,3), np.uint8)
    img=cv2.dilate(img,kernel,iterations = 1)
    ret,img=cv2.threshold(img,203,255,cv2.THRESH_BINARY)

    img=cv2.resize(img,[120,30])


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

print(predict('./test/32LPJ.png'))
