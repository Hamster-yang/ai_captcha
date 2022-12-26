import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import random

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Layer, BatchNormalization 
from keras.optimizers import Adam
from keras import Model, Input 
from keras.callbacks import EarlyStopping

import plotly.express as px
import plotly.graph_objects as go

import os

import string
captcha_list = []
img_shape = (30, 120, 1)
symbols = string.ascii_lowercase+ string.ascii_uppercase + "0123456789"

len_symbols = len(symbols) # the number of symbols
nSamples = len(os.listdir('./train')) # the number of samples 'captchas'
len_captcha = 5

X = np.zeros((nSamples, 30, 120, 1)) # 1070 * 50 * 200
y = np.zeros((5, nSamples, len_symbols)) # 5 * 1070 * 36

nSamples_test = len(os.listdir('./test')) # the number of samples 'captchas'

X_test1 = np.zeros((nSamples_test, 30, 120, 1)) # 1070 * 50 * 200
y_test1 = np.zeros((5, nSamples_test, len_symbols)) # 5 * 1070 * 36

for i, captcha in enumerate(os.listdir('./train')):
    captcha_code = captcha.split(".")[0]
    captcha_list.append(captcha_code)
    captcha_cv2 = cv2.imread(os.path.join('./train', captcha), cv2.IMREAD_GRAYSCALE)

        
    captcha_cv2=cv2.resize(captcha_cv2,[300,100])
    kernel = np.ones((3,3), np.uint8)
    captcha_cv2=cv2.dilate(captcha_cv2,kernel,iterations = 1)
    ret,captcha_cv2=cv2.threshold(captcha_cv2,203,255,cv2.THRESH_BINARY)

    captcha_cv2=cv2.resize(captcha_cv2,[120,30])
    
    captcha_cv2 = captcha_cv2 / 255.0
    captcha_cv2 = np.reshape(captcha_cv2, img_shape)
    targs = np.zeros((len_captcha, len_symbols))
    
    for a, b in enumerate(captcha_code):
        targs[a, symbols.index(b)] = 1
    
    X[i] = captcha_cv2
    y[:, i] = targs

for i, captcha in enumerate(os.listdir('./test')):
    captcha_code = captcha.split(".")[0]
    captcha_list.append(captcha_code)
    captcha_cv2 = cv2.imread(os.path.join('./test', captcha), cv2.IMREAD_GRAYSCALE)

        
    captcha_cv2=cv2.resize(captcha_cv2,[300,100])
    kernel = np.ones((3,3), np.uint8)
    captcha_cv2=cv2.dilate(captcha_cv2,kernel,iterations = 1)
    ret,captcha_cv2=cv2.threshold(captcha_cv2,203,255,cv2.THRESH_BINARY)

    captcha_cv2=cv2.resize(captcha_cv2,[120,30])
    
    captcha_cv2 = captcha_cv2 / 255.0
    captcha_cv2 = np.reshape(captcha_cv2, img_shape)
    targs = np.zeros((len_captcha, len_symbols))
    
    for a, b in enumerate(captcha_code):
        targs[a, symbols.index(b)] = 1
    
    X_test1[i] = captcha_cv2
    y_test1[:, i] = targs

print("shape of X:", X.shape)
print("shape of y:", y.shape)
print("shape of X_test1:", X_test1.shape)
print("shape of y_test1:", y_test1.shape)

X_train = X[:10000] 
y_train = y[:, :10000]
X_test = X_test1[:20000]
y_test = y_test1[:, :20000]

captcha = Input(shape=(30,120,1))
x = Conv2D(16, (3,3),padding='same',activation='relu')(captcha)
x = MaxPooling2D((2,2) , padding='same')(x)
x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)
x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)
x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)
x = BatchNormalization()(x)


flatOutput = Flatten()(x)

dense1 = Dense(64 , activation='relu')(flatOutput)
dropout1= Dropout(0.5)(dense1)
output1 = Dense(len_symbols , activation='sigmoid' , name='char_1')(dropout1)

dense2 = Dense(64 , activation='relu')(flatOutput)
dropout2= Dropout(0.5)(dense2)
output2 = Dense(len_symbols , activation='sigmoid' , name='char_2')(dropout2)
    
dense3 = Dense(64 , activation='relu')(flatOutput)
dropout3= Dropout(0.5)(dense3)
output3 = Dense(len_symbols , activation='sigmoid' , name='char_3')(dropout3)
    
dense4 = Dense(64 , activation='relu')(flatOutput)
dropout4= Dropout(0.5)(dense4)
output4 = Dense(len_symbols , activation='sigmoid' , name='char_4')(dropout4)
    
dense5 = Dense(64 , activation='relu')(flatOutput)
dropout5= Dropout(0.5)(dense5)
output5 = Dense(len_symbols , activation='sigmoid' , name='char_5')(dropout5)
    
model = Model(inputs = captcha , outputs=[output1 , output2 , output3 , output4 , output5])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
earlystopping = EarlyStopping(monitor ="val_loss",  
                             mode ="min", patience = 5,  
                             restore_best_weights = True) 

history = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=16, epochs=240, verbose=1, validation_split=0.2 )#callbacks =[earlystopping])

score = model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]],verbose=1)

print('Test Loss and accuracy:', score)
plt.figure(figsize=(15,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
def makePredict(captcha):
    captcha = np.reshape(captcha , (30,120))
    result = model.predict(np.reshape(captcha, (1,30,120,1)))
    result = np.reshape(result ,(5,62))
    indexes =[]
    for i in result:
        indexes.append(np.argmax(i))
        
    label=''
    for i in indexes:
        label += symbols[i]
        
    return label

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

image = cv2.imread("./test/OoqzD.png", cv2.IMREAD_GRAYSCALE)

# 使用自適應閾值法對圖像進行二值化
thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)

# 顯示原圖和二值化後的圖像
plt.imshow(image,cmap='gray')
plt.show()

plt.imshow(thresholded_image,cmap='gray')
plt.show()

image = cv2.imread("./test/QBsJp.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image,cmap='gray')
plt.show()
image=cv2.resize(image,[300,100])
kernel = np.ones((3,3), np.uint8)
image=cv2.dilate(image,kernel,iterations = 1)
ret,image=cv2.threshold(image,203,255,cv2.THRESH_BINARY)

image1=cv2.resize(image,[120,30])
plt.imshow(image1,cmap='gray')

plt.show()

while(1):
    str_path ="./test/" + input("path:")+ ".png"
    if str_path == './test/qqq.png':
        break
    str=predict(str_path)
    if str != '100000' :

        print(str)

model.save('model_v4-2_1.h5')