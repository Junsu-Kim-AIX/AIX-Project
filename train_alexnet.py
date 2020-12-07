import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l2
from collections import defaultdict
from glob import glob
import os
import random
import string
import logging
import time
from sklearn.model_selection import train_test_split
from cv2 import imread, IMREAD_GRAYSCALE
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Input
from keras.models import model_from_json

dir_name = '/home/esoc/cwlee/AIX/segments_aug/'
path = sorted(glob(dir_name+'/*'))
img_data = []
label_data = []

for i in range(np.size(path)):
	fname_folder = os.path.basename(path[i])
	path_img = os.path.join(dir_name, fname_folder)
	fname_img = glob(path_img+'/*.png')
	label_base = np.zeros((36))
	label_base[i] = 1
	for i in range(np.size(fname_img)):
		img = cv2.imread(fname_img[i], IMREAD_GRAYSCALE)
		img = cv2.resize(img, (64, 64))
		img_data.append(img)
		label_data.append(label_base)

img_data = np.expand_dims(np.array(img_data),3)/255
label_data = np.array(label_data)

#####################################################
def AlexNet(input_size, num_classes, summary=True):
	input_layer = Input(input_size)

# Layer 1
	conv1 = Conv2D(96, (11, 11), padding='same', strides=4,
	kernel_regularizer=l2(1e-4))(input_layer)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

# Layer 2
	conv2 = Conv2D(256, (5, 5), padding='same')(conv1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# Layer 3
	conv3 = Conv2D(384, (3, 3), padding='same')(conv2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)

# Layer 4
	conv4 = Conv2D(384, (3, 3), padding='same')(conv3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)

# Layer 5
	conv5 = Conv2D(256, (3, 3), padding='same')(conv4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)
	conv5 = MaxPooling2D(pool_size=(2, 2))(conv5)

# Layer 6
	conv6 = Flatten()(conv5)
	conv6 = Dense(4096)(conv6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Activation('relu')(conv6)
	conv6 = Dropout(0.5)(conv6)

# Layer 7
	conv7 = Dense(4096)(conv6)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation('relu')(conv7)
	conv7 = Dropout(0.5)(conv7)

# Layer 8
	conv8 = Dense(num_classes)(conv7)
	conv8 = BatchNormalization()(conv8)
	conv8 = Activation('softmax')(conv8)

	output_layer = conv8

	model = keras.Model(inputs=[input_layer], outputs=[output_layer])
	model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
	if summary:
		model.summary()

	return model

input_size = [64, 64, 1]
num_classes = 36

model = AlexNet(input_size, num_classes=num_classes)
#####################################################

X_train, X_valid, Y_train, Y_valid = train_test_split(img_data, label_data, test_size=0.2)
# creating the final model

MODEL_SAVE_FOLDER_PATH = '/home/esoc/cwlee/AIX/alex_model'
model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

t0= time.time()
hist = model.fit(X_train, Y_train, batch_size=64, epochs=50, verbose=1, validation_data=(X_valid, Y_valid), callbacks=[checkpoint])
t1= time.time()

exe_time=t1-t0

with open('/home/esoc/cwlee/AIX/trainHistoryDict_Alex', 'wb') as file_pi:
	pickle.dump(hist.history, file_pi)

plt.figure(1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss

plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("execution time: ", str(exe_time)+" s")
