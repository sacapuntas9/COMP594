#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from time import time
from keras.callbacks import ModelCheckpoint
import cv2
import random, os
import numpy as np

directorySeparator = "\\"

testDir = "E:\\594_data\\custom_NN_Data\\test_holdout\\xd"
outputDir = "E:\\594_data\\custom_NN_Data\\test_holdout\\classified_both" 
window_size = 65
middle = int(window_size / 2)


# In[2]:


def mirror_image(img):
    height, width = img.shape

    #the following creates a mirrored image with the edges mirrored using a size window_size

    newImg = np.zeros((height+(window_size*2),width+(window_size*2)), np.uint8)
    for h in range(height):
        for w in range(width):
            newImg[h+window_size,w+window_size] = img[h,w]

    for w in range(window_size):
        for h in range(height):
            newImg[h+window_size,w]=img[h,window_size-1-w]
            newImg[h+window_size,width+(2*window_size)-1-w]=img[h,width-(window_size-w)]

    for h in range(window_size):
        for w in range(width):
            newImg[h,w+window_size]=img[window_size-1-h,w]
            newImg[height+(2*window_size)-1-h,w+window_size]=img[height-(window_size-h),w]
            
    return newImg
    
                
    
    


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(1,65,65 ), data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.load_weights("weights.best.hdf5")
print("loaded weights from disk")

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

test_array = []

for filename in os.listdir(testDir):
    if filename.endswith(".tif"): 
        img = cv2.imread(testDir+directorySeparator+filename,-1)
        img = (img/256).astype('uint8')
        mirror = mirror_image(img)
    
        height,width = img.shape
        
        for h in range(height):
            for w in range(width):
                crop_img = mirror[(h+window_size-middle):(h+window_size+middle+1), (w+window_size-middle):(w+window_size+middle+1)]
                test_array.append([crop_img / (1./255)])
        
        test_array = np.array(test_array)
        
        print(test_array.shape)
        
        
prediction = model.predict(test_array)

print(prediction)


# In[ ]:




