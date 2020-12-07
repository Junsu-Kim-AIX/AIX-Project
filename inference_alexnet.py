import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import tensorflow as tf
import keras
from keras import applications
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input
from keras.regularizers import l2
from keras.utils import np_utils
from collections import defaultdict
from glob import glob
import os
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from cv2 import imread, IMREAD_GRAYSCALE
from collections import defaultdict
from keras.models import model_from_json

X_test= pickle.load(open('/home/esoc/cwlee/AIX/test_data.p', 'rb'))
print('Type of X_test : {}'.format(type(X_test)))
print('Shape of X_test : {}'.format(X_test.shape))


y_test=['56m6y', '7m8px', '46mbm', 'mfb3x', 'p5nce', '865wm', 'g842c', 'nbwnn', 'ydd3g', 'fncnb', 'xc68n', '6cm6m', '3ndxd', 'w75w8', '8cm46', 'c3n8x', 'gcfgp','bc8nf', 'b28g8', 'wxcn8', '677g3', 'fc6xb', 'ybfx6', 'dnmd8', 'cpc8c', 'mmfm6', 'yew6p', '5np4m', 'xdcn4', '4m2w5', 'c2yn8', '5wddw', 'nbmx7', 'fg38b', '7fde7','de45x', 'e4gd7', 'mgw3n', '56ncx', 'm448b', '373gb', '8npe3', '25w53', '24f6w', 'edg3p', 'wgmwp', '2356g', '33n73', '8w754', 'ffnxn', 'dy3cx', '226md', '5fyem', '62fgn', 'bpwd7', 'd3c7y', '3den6', '8np22', '7cgym', 'dgcm4', '3bnd7', '8c23f', 'b4ncn', 'mpmy5', 'gegw4', 'bbymy', '76353', '6wg4n', '77387', '537nf']

num_test_samples = len(X_test)
X_test_pre=[] #Preprocessed X_train set

for n in range(num_test_samples):

	ret,thresh1 = cv2.threshold(X_test[n],127,255,cv2.THRESH_BINARY_INV) #Otsu's segment
	kernel = np.ones((3,3),np.uint8) 
	opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel) # Morphological opening
	X_test_pre.append(opening)

intensity = 0 
intensity_array = []
len_column = len(X_test[0][0])
len_row = len(X_test[0])

for n in range(num_test_samples):
	for k in range (len_column):
		for m in range(len_row):
			intensity += X_test_pre[n][m,k]

		if(intensity > 255*15):
			intensity_array.append(k)

		intensity = 0

	col1 = min(intensity_array)
	col2 = max(intensity_array) + 5

	X_test_pre[n] = X_test_pre[n][:, col1: col2]
	del intensity_array[:]
###############################################

def AlexNet(input_size, num_classes, summary=True):
	input_layer = Input(input_size)

# Layer 1
	conv1 = Conv2D(96, (11, 11), padding='same', strides=4, kernel_regularizer=l2(1e-4))(input_layer)
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
	model.load_weights("alex_latest_model.hdf5")
	print("Loaded model from disk")

	model.compile(optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
	
	return model

input_size = [64, 64, 1]
num_classes = 36
model = AlexNet(input_size, num_classes=num_classes)
model.load_weights("alex_latest_model.hdf5")
print("Loaded model from disk")
	
	
#############################################

def map_func(y_array):
	m = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
	for num1 in range(36):
		if(y_array[0][num1]==max(y_array[0])):
			r = m[num1]
	return r
##############################################
y_label=""
y_test_pred=[]
accuracy = 0
for n in range(50):
	for num in range(5):
		x_leng = len(X_test_pre[n][0])
		x1 = (int)(num*x_leng/5)
		x2 = (int)((num+1)*x_leng/5)
		img = X_test_pre[n][:, x1:x2]
		img = cv2.resize(img,(64, 64), interpolation=cv2.INTER_CUBIC)
		img = img[np.newaxis]
		img = np.expand_dims(np.array(img),3)/255
    #img = np.concatenate([img, img, img], axis =3)
		y = model.predict(img)
		y_label = y_label + map_func(y)
	y_test_pred.append(y_label)
	y_label=""

print(y_test_pred)

count = 0
for n in range(50):
	for m in range(5):
		if(y_test_pred[n][m]!=y_test[n][m]):
			count=count+1

accuracy = 100 - (count/250 * 100)
print("Test Accuracy = ",accuracy,"%")
#########################################################
