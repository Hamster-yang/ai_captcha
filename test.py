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
import cv2 as cv
import string
import random
import glob
import string
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
def find_squares(filepath):
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    squares = []
    img_org = img
    img = cv.GaussianBlur(img, (3, 3), 0)   
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bin = cv.Canny(gray, 30, 100, apertureSize=3)    
    contours, _hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print("輪廓数量：%d" % len(contours))
    index = 0
    # 輪廓
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True) #計算輪廓周長
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True) #多邊形逼近
        # 条件判断逼近边的数量是否为4，輪廓面积是否大于7000，检测轮廓是否为凸的
        if len(cnt) == 4 and cv.contourArea(cnt) > 2000 and cv.isContourConvex(cnt):
            print("in")
            M = cv.moments(cnt) #计算轮廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#輪廓数量
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            '''
            print(cnt[0])
            print(cnt[1])
            print(cnt[2])
            print(cnt[3])
            '''
            
            x = cnt[0][0]
            y = cnt[0][1]

            # 裁切區域的長度與寬度
            w = 120
            h = 30

            # 裁切圖片 
            
            crop_img = img_org[y:y+h, x:x+w]
            find = True
            # 只检测矩形（cos90° = 0）
            #if max_cos < 0.1:
            # 检测四边形（不限定角度范围）
            '''
            if True:
                index = index + 1
                cv.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                squares.append(cnt)
            '''
            return crop_img, squares, img, find
        else:
            find = False
            return img, squares, img, find
def main():
    
    output_folder = './output/'
      
    crop_img, squares, img = find_squares(".output/full_screen.png")
    print(crop_img)


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()        