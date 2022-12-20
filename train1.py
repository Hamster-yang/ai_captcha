# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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

#Init main values
symbols = string.ascii_lowercase+ string.ascii_uppercase + "0123456789" # All symbols captcha can contain#-------------------------------------------------------
num_symbols = len(symbols)
img_shape = (50,200 , 1)#-------------------------------------------------------
 
#print(num_symbols)


def create_model():
    img = layers.Input(shape=img_shape) # Get image as an input and process it through some Convs
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 100x25
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 50x13
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)  # 25x7
    
    # Get flattened vector and make 5 branches from it. Each branch will predict one letter
    flat = layers.Flatten()(mp3)
    outs = []
    for _ in range(5):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        res = layers.Dense(num_symbols, activation='sigmoid')(drop)

        outs.append(res)
    
    # Compile model and return it
    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    return model

def preprocess_data():
    n_samples = len(os.listdir('./train'))
    X = np.zeros((n_samples, 50, 200, 1)) #1070*50*200#-------------------------------------------------------
    y = np.zeros((5, n_samples, num_symbols)) #5*1070*36

    for i, pic in enumerate(os.listdir('./train')):
        # Read image as grayscale
        img = cv2.imread(os.path.join('./train', pic), cv2.IMREAD_GRAYSCALE)
        
        pic_target = pic[:-4]
        if len(pic_target) < 6:
            # Scale and reshape image
            img = img / 255.0
            img = np.reshape(img, (50, 200, 1))#-------------------------------------------------------
            # Define targets and code them using OneHotEncoding
            targs = np.zeros((5, num_symbols))
            for j, l in enumerate(pic_target):
                ind = symbols.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs
    
    # Return final data
    return X, y

X, y = preprocess_data()
'''
state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(y)'''

X_train, y_train = X[:10000], y[:, :10000]#-------------------------------------------------------
X_test, y_test = X[1000:2000], y[:, 1000:2000]

model=create_model()
#model.summary()

hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30,verbose=2, validation_split=0.2)#-------------------------------------------------------


# Define function to predict captcha
def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
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

score= model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]],verbose=2)
print('Test Loss and accuracy:', score)



# Check model on some samples
model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])

#model.save('model.h5')

print(predict('C:/Users/allen/Desktop/ai_captcha/train/0A6JM.png'))
print(predict('C:/Users/allen/Desktop/ai_captcha/train/JTtbn.png'))
print(predict('C:/Users/allen/Desktop/ai_captcha/train/LoORS.png'))
print(predict('C:/Users/allen/Desktop/ai_captcha/train/90WYy.png'))
print(predict('C:/Users/allen/Desktop/ai_captcha/train/nP59g.png'))
print(predict('C:/Users/allen/Desktop/ai_captcha/train/Ulg0R.png'))
'''print("HAHAHAHAHAHAHAHAHAHAHAHAHAHAHAHAHAH")
print(predict('C:/Users/allen/Desktop/ai_captcha/train/0A6JM.png'))
print(predict('C:/Users/allen/Desktop/ai_ml_opencv_pi/test/3cHfy.png'))
print(predict('C:/Users/allen/Desktop/ai_ml_opencv_pi/test/4sYrK.png'))
print(predict('C:/Users/allen/Desktop/ai_ml_opencv_pi/test/YYWB8.png'))
print(predict('C:/Users/allen/Desktop/ai_ml_opencv_pi/test/z7JUi.png'))
print(predict('C:/Users/allen/Desktop/ai_ml_opencv_pi/test/xCjjJ.png'))
print(predict('C:/Users/allen/Desktop/ai_ml_opencv_pi/test/6xx7X.png'))
print(predict('C:/Users/allen/Desktop/ai_ml_opencv_pi/test/3e8Co.png'))
'''

model.save('model.h5')

while(1):
    str_path = input("path:")
    str=predict(str_path)
    if str != '100000' :
        print(str)

#print(predict('../input/captcha-version-2-images/samples/samples/8n5p3.png'))

#Lets test an unknown captcha
#preview
#import matplotlib.pyplot as plt
#img=cv2.imread('./test/'+test_y[0]+'.jpg',cv2.IMREAD_GRAYSCALE)
#plt.imshow(img, cmap=plt.get_cmap('gray'))

#Lets Predict By Model
#print("Predicted Captcha =",predict('./test'+test_y[0]+'.jpg'))

