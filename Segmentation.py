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


X_train, y_train = pickle.load(open('/home/esoc/cwlee/AIX/train_data.p', 'rb'))

print('Type of X_train : {}'.format(type(X_train)))
print('Shape of X_train : {}'.format(X_train.shape))
print('Type of y_train : {}'.format(type(y_train)))
print('Length of y_train : {}'.format(len(y_train)))

X_test= pickle.load(open('/home/esoc/cwlee/AIX/test_data.p', 'rb'))
print('Type of X_test : {}'.format(type(X_test)))
print('Shape of X_test : {}'.format(X_test.shape))





num_train_samples = len(X_train)
X_train_pre=[] #Preprocessed X_train set

for n in range(num_train_samples):

	ret,thresh1 = cv2.threshold(X_train[n],127,255,cv2.THRESH_BINARY_INV) #Otsu's segment
	kernel = np.ones((3,3),np.uint8) 
	opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel) # Morphological opening
	X_train_pre.append(opening)

intensity = 0 
intensity_array = []
len_column = len(X_train[0][0])
len_row = len(X_train[0])

for n in range(num_train_samples):
	for k in range (len_column):
		for m in range(len_row):
			intensity += X_train_pre[n][m,k]

		if(intensity > 255*15):
			intensity_array.append(k)

		intensity = 0

	col1 = min(intensity_array)
	col2 = max(intensity_array) + 5

	X_train_pre[n] = X_train_pre[n][:, col1: col2]
	del intensity_array[:]
###############################################

raw_path ='/home/esoc/cwlee/AIX/samples/'
seg_path = '/home/esoc/cwlee/AIX/segments/'
counts = defaultdict(int)
num_train_samples = len(X_train_pre)

for n in range(1000):
	for num in range(5):
		x_leng = len(X_train_pre[n][0])
		x1 = (int)(num*x_leng/5)
		x2 = (int)((num+1)*x_leng/5)


		img = X_train_pre[n][:, x1:x2]

		letter = y_train[n][num]

		filename = "/" + str(counts[letter]).zfill(5) + ".png"

		path = seg_path + letter + filename
		cv2.imwrite(path, img)
		counts[letter] += 1
	
	
print("done with training samples")


#########################################################
