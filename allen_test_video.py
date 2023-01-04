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
import cv2
import string
import random
import glob
import string
captcha_list = []
img_shape = (30, 120, 1)
symbols = string.ascii_lowercase+ string.ascii_uppercase + "0123456789"

len_symbols = len(symbols) # the number of symbols
nSamples = len(os.listdir('train')) # the number of samples 'captchas'
len_captcha = 5

model=load_model('model_v4_5.h5')


def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    img=cv2.resize(img,[300,100])
    kernel = np.ones((3,3), np.uint8)
    img=cv2.dilate(img,kernel,iterations = 1)
    ret,img=cv2.threshold(img,220,255,cv2.THRESH_BINARY)

    img=cv2.resize(img,[120,30])
    cv2.imshow("321",img)
    cv2.waitKey()

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


# 设置putText函数字体
font=cv2.FONT_HERSHEY_SIMPLEX
#计算两边夹角额cos值
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img_org):
    squares = []
    img = cv2.GaussianBlur(img_org, (3, 3), 0)   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin = cv2.Canny(gray, 30, 100, apertureSize=3)    
    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("輪廓数量：%d" % len(contours))
    index = 0
    # 輪廓
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True) #計算輪廓周長
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #多邊形逼近
        #print(cnt_len)
        # 条件判断逼近边的数量是否为4，輪廓面积是否大于7000，檢测輪廓是否為凸的
        if len(cnt) == 4 and cv2.contourArea(cnt) > 700 and cv2.isContourConvex(cnt):
            M = cv2.moments(cnt) #計算輪廓的矩
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
            w = 120
            h = 30

            # 裁切圖片
            crop_img = img_org[y:y+h, x:x+w]

            # 只檢测矩形（cos90° = 0）
            #if max_cos < 0.1:
            # 檢测四邊形（不限定角度范圍）
            '''
            if True:
                index = index + 1
                cv.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                squares.append(cnt)
            '''
    return crop_img, squares, img



def main():

    video_path = 'testvideo_7s_5pic.mp4'
    output_folder = './output/'

    '''if os.path.isdir(output_folder):
        print("Delete old result folder: {}".format(output_folder))
        #os.system("rimraf {}".format(output_folder))
    os.system("mkdir {}".format(output_folder))
    print("create folder: {}".format(output_folder))
'''
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    video = []

    for idx in range(frame_count):
        vc.set(1, idx)
        ret, frame = vc.read()
        height, width, layers = frame.shape
        size = (width, height)
        if idx%30==0:
            video.append(frame)
            #if frame is not None:
                #file_name = '{}{:08d}.jpg'.format(output_folder,idx)
                #cv2.imwrite(file_name, frame)
        

        print("\rprocess: {}/{}".format(idx+1 , frame_count), end = '')
    vc.release()

    for i in range(int(frame_count/30)):
      #img = cv2.imread("./ai_captcha/image.png")  
      img = video[i]       
      crop_img, squares, img = find_squares(img)
      cv2.drawContours( img, squares, -1, (0, 0, 255), 2 )
      file_name = '{}0000.png'.format(output_folder)
      cv2.imwrite(file_name, crop_img)
      img_predict = cv2.imread('{}0000.png'.format(output_folder))

      
      print(predict(file_name))
      cv2.imshow("123",crop_img)
      cv2.waitKey()
      # 顯示圖片
    
      print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
####################################################################    
