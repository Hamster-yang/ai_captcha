from PIL import ImageGrab
import numpy as np
import cv2 as cv
# 设置putText函数字体
font=cv.FONT_HERSHEY_SIMPLEX
#计算两边夹角额cos值
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    squares = []
    img = cv.GaussianBlur(img, (3, 3), 0)   
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bin = cv.Canny(gray, 30, 100, apertureSize=3)    
    contours, _hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("輪廓数量：%d" % len(contours))
    index = 0
    # 輪廓
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True) #計算輪廓周長
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True) #多邊形逼近
        # 条件判断逼近边的数量是否为4，輪廓面积是否大于1000，检测轮廓是否为凸的
        if len(cnt) == 4 and cv.contourArea(cnt) > 7000 and cv.isContourConvex(cnt):
            M = cv.moments(cnt) #计算轮廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#輪廓数量
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            print(cnt[0])
            print(cnt[1])
            print(cnt[2])
            print(cnt[3])
            x = cnt[0][0]
            y = cnt[0][1]

            # 裁切區域的長度與寬度
            w = 200
            h = 50

            # 裁切圖片
            crop_img = img[y:y+h, x:x+w]

            # 只检测矩形（cos90° = 0）
            #if max_cos < 0.1:
            # 检测四边形（不限定角度范围）
            '''
            if True:
                index = index + 1
                cv.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                squares.append(cnt)
            '''
    return crop_img, squares, img

def main():
    while True:
        # 全螢幕擷取
        img_rgb = ImageGrab.grab()
        img_bgr = cv.cvtColor(np.array(img_rgb), cv.COLOR_RGB2BGR)
        cv.imshow('imm', img_bgr)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv.destroyAllWindows()

    
    img = cv.imread("./ai_captcha/image.png")
            
    crop_img, squares, img = find_squares(img)
    cv.drawContours( img, squares, -1, (0, 0, 255), 2 )
    cv.imshow('squares', img) 
    cv.imshow("cropped", crop_img)
    cv.imwrite('result.jpg', crop_img)
    ch = cv.waitKey()

    # 顯示圖片
    


    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()