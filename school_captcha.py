from PIL import ImageGrab
import numpy as np
import cv2 as cv

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
import string


# 設置putText函数字體
font=cv.FONT_HERSHEY_SIMPLEX
#兩邊夾角cos計算值

captcha_list = []
img_shape = (30, 120, 1)
symbols = string.ascii_lowercase+ string.ascii_uppercase + "0123456789"

len_symbols = len(symbols) # the number of symbols
nSamples = len(os.listdir('train')) # the number of samples 'captchas'
len_captcha = 5

model=load_model('model_v4_5.h5')


def predict(filepath):
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    
    '''output_folder = './output/'
    cv.imshow("321",img)
    file_name = '{}321.png'.format(output_folder)
    cv.imwrite(file_name, img)'''
    
    img=cv.resize(img,[300,100])
    kernel = np.ones((3,3), np.uint8)
    img=cv.dilate(img,kernel,iterations = 1)
    ret,img=cv.threshold(img,203,255,cv.THRESH_BINARY)

    img=cv.resize(img,[120,30])

    if img is not None:
        img = img / 255.0
    else:
        print("Not detected")
        
        return '100000'
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis],verbose=0))
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




def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img_org):
    squares = []
    img = cv.GaussianBlur(img_org, (3, 3), 0)   
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    bin = cv.Canny(gray, 30, 100, apertureSize=3) 
    
    
    contours, _hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print("輪廓数量：%d" % len(contours))
    index = 0
    # 輪廓
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True) #計算輪廓周長
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True) #多邊形逼近
        #print(cnt_len)
        # 條件判斷逼近邊的數量是否為4，輪廓面積是否大於5600，檢測輪廓是否為凸的
        if len(cnt) == 4 and cv.contourArea(cnt) > 5600 and cv.contourArea(cnt) < 5650  and cv.isContourConvex(cnt): #and cv.contourArea(cnt) < 5000  len(cnt) == 4 and   and cv.isContourConvex(cnt)
            
            M = cv.moments(cnt) #計算輪廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#輪廓數量
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
            w = 150
            h = 37

            # 裁切圖片 
            
            crop_img = img_org[y-1:y+h, x+1:x+w]
            crop_img=cv.resize(crop_img,[120,30])
            #kernel = np.ones((3,3), np.uint8)
            #crop_img=cv.erode(crop_img,kernel,iterations = 1)
            find = True
            # 只檢測矩形（cos90° = 0）
            #if max_cos < 0.1:
            # 檢測四邊形（不限定角度範圍）
            '''
            if True:
                index = index + 1
                cv.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                squares.append(cnt)
            '''
            return crop_img, squares, img, find
    find = False
    return img, squares, img, find

def main():
    ago_filename=''
    output_folder = './output/'
    while True:
        # 全螢幕擷取
        img_rgb = ImageGrab.grab()
        img_bgr = cv.cvtColor(np.array(img_rgb), cv.COLOR_RGB2BGR)
        #show_img_1=cv.resize(img_bgr, (540, 540))  
        cv.namedWindow("imm", cv.WINDOW_NORMAL)
        cv.imshow('imm', img_bgr)
        name = '{}full_screen.png'.format(output_folder)
        cv.imwrite(name, img_bgr)
        crop_img, squares, img, find = find_squares(img_bgr)
        if find == True:
            
            file_name = '{}test.png'.format(output_folder)
            cv.imwrite(file_name, crop_img)
            
            captcha_name=predict(file_name)
            
            
            if ago_filename== captcha_name:
                ago_filename=captcha_name
            else:    
                print(captcha_name)
                ago_filename=captcha_name
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv.destroyAllWindows()
    # 顯示圖片
    


    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()