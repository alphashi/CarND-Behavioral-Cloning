import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
import scipy
from keras.layers import Convolution2D, ELU, Flatten, Dense, Dropout, Lambda, Input

import tensorflow as tf
tf.python.control_flow_ops = tf

EPOCHS = 5
BATCH_SIZE = 128

CAMERA_C_IDX = 0
CAMERA_L_IDX = 1
CAMERA_R_IDX = 2
ANGLE_IDX = 3

DATASET = "./dataset"

# only center camera is used
GEN_MODE_CENTER = 0
# center camera + augmented left and right cameras are used
GEN_MODE_ALL = 1
# stich view from all cameras and sample many views with augmented angle
GEN_MODE_GEN = 2

image_columns = 200
image_rows = 66
image_channels = 3

def normalize_input(x):
    return x / 255. - 0.5

def resize_input(x):
    cropped = x[66:152, 30:290,:]
    resized = cv2.resize(cropped, (200,66))
    return resized

def preproccess(x):
    img = resize_input(x)
    img = normalize_input(img)
    return img

def passthrough(x):
    return x

def augment_angle(center, left, right, angle):
    return []

def augment_generate(center, left, right, angle):
    return []

def data_generator(data, mode = GEN_MODE_CENTER, batch_size = 32, preprocess_input=normalize_input, shuffle = True):
    i = 0
    x_buffer, y_buffer, buffer_size = [], [], 0

    while True:
        i = i % len(data)
        item = data.loc[i]

        angle = item.get('steering')
        center = item.get('center')
        left = item.get('left')
        right = item.get('right')

        if mode == GEN_MODE_CENTER:
            path = os.path.join(DATASET, center)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_buffer.append(img)
            y_buffer.append(angle)

        elif mode == GEN_MODE_ALL:
            img, ang = augment_angle(center, left, right, angle)
            [x_buffer.append(im) for im in img]
            [y_buffer.append(an) for an in angle]

        elif mode == GEN_MODE_GEN:
            img, ang = augment_generate(center, left, right, angle)
            [x_buffer.append(im) for im in img]
            [y_buffer.append(an) for an in angle]

        buffer_size = len(x_buffer)

        if buffer_size > 0:
            x_b = np.stack(x_buffer, axis=0)
            y_b = np.stack(y_buffer, axis=0)
            for x, y in zip(x_b, y_b):
                x = preprocess_input(x)
                x = np.expand_dims(x, axis=0)
                y = np.expand_dims(y, axis=0)
                yield x, y

def nvidia_model():
    input_shape = (image_rows, image_columns, image_channels)
    model = Sequential()
    #model.add(Lambda(lambda x: x/127.5 - 1.,input_shape = input_shape))
    model.add(Convolution2D(24, 5, 5, input_shape=input_shape, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
    model.add(Flatten())
    model.add(ELU())
    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal'))
    return model

def unknown_model():
    input_shape = (image_rows, image_columns, image_channels)
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape=input_shape, subsample=(2, 2),
                            border_mode='valid',
                            name='conv1', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2),
                            border_mode='valid',
                            name='conv2', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2),
                            border_mode='valid',
                            name='conv3', init='he_normal'))
    model.add(ELU())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1),
                            border_mode='valid',
                            name='conv4', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1),
                            border_mode='valid',
                            name='conv5', init='he_normal'))
    model.add(ELU())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, name='hidden1', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(50, name='hidden2', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(10, name='hidden3', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1, name='output', init='he_normal'))
    return model


if __name__ == '__main__':

    data = pd.read_csv(os.path.join(DATASET, "driving_log.csv"))

    train_genereator = data_generator(data, GEN_MODE_CENTER, preprocess_input=preproccess)

    model = nvidia_model()
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')
    #if weights_path:
    #    model.load_weights(weights_path)

    #model.summary()

    model.fit_generator(train_genereator, samples_per_epoch=len(data), nb_epoch=4)

    # x,y = next(train_genereator)
    # plt.figure(figsize=(32, 8))
    # a = x[0].shape
    # plt.imshow(x[0])
    # plt.show()

    print("End")

