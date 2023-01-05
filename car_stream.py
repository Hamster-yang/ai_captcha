#t.ly/rbXh
import darknet
import time
import cv2

TARGET= 'https://trafficvideo2.tainan.gov.tw/e452449h'
WEIGHT="yolov4.weights"
CFG="./cfg/yolov4.cfg"
DATA="./cfg/coco.data"
CONFTH=0.5
(WIDTH,HEIGHT)=(640,640) 

network, class_names, class_colors = darknet.load_network(
    CFG,
    DATA,
    WEIGHT,
)

#Darknet YOLO辨識主程式
def image_detection(image, network, class_names, class_colors, thresh):

    darknet_image = darknet.make_image(WIDTH, HEIGHT, 3)  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (WIDTH, HEIGHT))

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


cap = cv2.VideoCapture(TARGET)
while 1:
    ret, frame = cap.read() #ret=retval,frame=image
    stime = time.time()
    #                                   影像   神經網路   物件名稱      顏色         信任度
    frame, detections = image_detection(frame, network, class_names, class_colors, CONFTH  )
    darknet.print_detections(detections)
    # 計算車流量
    # PredDict={'car':0,'truck':0,'person':0,'bus':0,'motorbike':0}
    # for i in range(len(detections)):
    #     if detections[i][0] in PredDict.keys():
    #         PredDict[detections[i][0]] += 1
    # print(PredDict)


    cv2.imshow('Inference', frame)
    key =  cv2.waitKey(1)
    if key == ord('q'):
        break
    etime = time.time()
    fps = round(1/(etime-stime),3)
    print("FPS: {}".format(fps))


