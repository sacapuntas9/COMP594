#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from time import time
from keras.callbacks import ModelCheckpoint

trainDir = "E:\\594_data\\custom_NN_Data\\train"
validDir = "E:\\594_data\\custom_NN_Data\\valid"


# In[3]:


batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        data_format='channels_first',
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255,         
                                  data_format='channels_first'
                                 )

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        trainDir,  # this is the target directory
        target_size=(65, 65),  # all images will be resized
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validDir,
        target_size=(65, 65),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary')


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


checkpoint = ModelCheckpoint("weights.best.hdf5", verbose=1, monitor='val_acc',save_best_only=True, mode='auto')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['binary_accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=1000 // batch_size,
        verbose=1, 
        callbacks=[checkpoint,tensorboard])


# In[ ]:




