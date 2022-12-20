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

