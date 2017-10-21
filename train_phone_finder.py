from __future__ import division, print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import numpy as np
from skimage.io import imread
import sys, os
from keras import backend as K
import random
w = 326
h = 490
d = 3
epoch = 35
data = []
train_dataset = sys.argv[1]
if train_dataset.find('.jpg') > -1:
    train_dataset = 'find_phone_task/find_phone'

with open(os.path.join(train_dataset, 'labels.txt'), 'rb') as fp:
    for lines in fp.readlines():
        file_name, _x, _y = lines.split()
        img = imread(os.path.join(train_dataset, file_name))
        data.append((img, np.array([float(_x), float(_y)])))
random.shuffle(data)
train_data = data[:115]
#dev_data = data[99:115]
test_data = data[115:]

def get_data(dat):
    x = [x for x,_ in dat]
    y = [y for _,y in dat]
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def proximity_loss(pred, target):
    loss = 0.0
    for i in range(len(pred)):
        dist = np.sqrt((pred[0] - target[0])**2 + (pred[1] - target[1])**2)
        loss += 1 if dist < 0.05 else 0
    return np.mean(loss)



model = Sequential()
model.add(Convolution2D(32, (3, 3),  input_shape=(w, h, d)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Convolution2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(lr=0.0001))

_x, _y = get_data(data)
_x = _x / np.sum(_x)*255.0#keras.utils.normalize(_x) # * np.max(_x)/ 255
model.fit(_x, _y, validation_split=0.10, epochs=epoch, batch_size=29, shuffle=True)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('weights.h5')



