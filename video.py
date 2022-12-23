'''import cv2
# 開啟影片檔案
cap = cv2.VideoCapture('C:/Users/allen/Desktop/ai_captcha/20221219-1.avi')
# 以迴圈從影片檔案讀取影格，並顯示出來
while(cap.isOpened()):
  ret, frame = cap.read()
  cv2.imshow('frame',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
'''

import os
import cv2
import glob
import numpy as np 

'''
video_path = 'testvideo.mp4'
output_folder = './'
if os.path.isdir(output_folder):
    print("Delete old result folder: {}".format(output_folder))
    #os.system("rimraf {}".format(output_folder))
os.system("mkdir {}".format(output_folder))
print("create folder: {}".format(output_folder))
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
        if frame is not None:
            file_name = '{}{:08d}.jpg'.format(output_folder,idx)
            cv2.imwrite(file_name, frame)
    
    print("\rprocess: {}/{}".format(idx+1 , frame_count), end = '')
vc.release()
for i in range(70):
  cv2.imshow('frame',video[i])
  cv2.waitKey()
____________'''

# 设置putText函数字体
font=cv2.FONT_HERSHEY_SIMPLEX
#计算两边夹角额cos值
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    squares = []
    img = cv2.GaussianBlur(img, (3, 3), 0)   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin = cv2.Canny(gray, 30, 100, apertureSize=3)    
    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("輪廓数量：%d" % len(contours))
    index = 0
    crop_img1 = None
    crop_img1 = img[10, 100]
    # 輪廓
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True) #計算輪廓周長
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #多邊形逼近
        # 条件判断逼近边的数量是否为4，輪廓面积是否大于7000，檢测輪廓是否為凸的
        if len(cnt) == 4 and cv2.contourArea(cnt) > 700 and cv2.isContourConvex(cnt):
            M = cv2.moments(cnt) #計算輪廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#輪廓數量
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            print(cnt[0])
            print(cnt[1])
            print(cnt[2])
            print(cnt[3])
            x = cnt[0][0]
            y = cnt[0][1]

            # 裁切區域的長度與寬度
            w = 120
            h = 30
            
            # 裁切圖片
            crop_img1 = img[y:y+h, x:x+w]

            # 只檢测矩形（cos90° = 0）
            #if max_cos < 0.1:
            # 檢测四邊形（不限定角度范圍）
            '''
            if True:
                index = index + 1
                cv.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                squares.append(cnt)
            '''
    return crop_img1, squares, img

def main():
    crop_img = ''
    video_path = './testvideo12s5pic.mp4'
    output_folder = './output/'

    if os.path.isdir(output_folder):
        print("Delete old result folder: {}".format(output_folder))
        #os.system("rimraf {}".format(output_folder))
    os.system("mkdir {}".format(output_folder))
    print("create folder: {}".format(output_folder))

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

    for i in range(70):
        cv2.waitKey()
###########################################################
        #img = cv2.imread("./ai_captcha/image.png")  
        img = video[i]       
        crop_img, squares, img = find_squares(img)
        cv2.drawContours( img, squares, -1, (0, 0, 255), 2 )
        cv2.imshow("cropped", crop_img)
        file_name = '{}{:04d}.jpg'.format(output_folder,i)
        #cv2.imwrite(file_name, crop_img)
        ch = cv2.waitKey()

        # 顯示圖片
        


        print('Done')



if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()