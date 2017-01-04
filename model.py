import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, ELU, Flatten, Dense, Dropout, Lambda, Input

import tensorflow as tf
tf.python.control_flow_ops = tf

EPOCHS = 10
BATCH_SIZE = 256

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


def augment_angle(center, left, right, angle, shift):
    images, angles = [], []
    images.append(center)
    angles.append(angle)
    images.append(left)
    angles.append(angle + shift)
    images.append(right)
    angles.append(angle - shift)
    return images, angles


def augment_generate(center, left, right, angle):
    return []


def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (image_columns, image_rows))

    return image_tr, steer_ang


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def data_generator(data, mode = GEN_MODE_CENTER, batch_size = 32, preprocess_input=normalize_input, shuffle = True):
    i = 0
    x_buffer, y_buffer, buffer_size = [], [], 0

    while True:
        i = i % len(data)
        item = data.loc[i]

        angle = item.get('steering')
        center = item.get('center').strip()
        left = item.get('left').strip()
        right = item.get('right').strip()

        # read images and generate augmented images into the buffer

        if mode == GEN_MODE_CENTER:
            path = os.path.join(DATASET, center)
            img = read_image(path)
            x_buffer.append(img)
            y_buffer.append(angle)

        elif mode == GEN_MODE_ALL:
            center_img = read_image(os.path.join(DATASET, center))
            left_img = read_image(os.path.join(DATASET, left))
            right_img = read_image(os.path.join(DATASET, right))
            img, ang = augment_angle(center_img, left_img, right_img, angle, 0.05)
            [x_buffer.append(im) for im in img]
            [y_buffer.append(an) for an in ang]

        elif mode == GEN_MODE_GEN:
            img, ang = augment_generate(center, left, right, angle)
            [x_buffer.append(im) for im in img]
            [y_buffer.append(an) for an in ang]

        buffer_size = len(x_buffer)

        # drain the buffer

        if buffer_size >= batch_size:
            num_batches = int(buffer_size / batch_size)
            head = num_batches * batch_size
            tail = buffer_size - head
            indx = list(range(head))
            if shuffle:
                np.random.shuffle(indx)

            preprocessed_x = [preprocess_input(im) for im in x_buffer[:head]]
            x = np.stack(preprocessed_x, axis=0)
            y = np.stack(y_buffer[:head], axis=0)

            for b in range(num_batches):
                start = b * batch_size
                end = (b+1) * batch_size
                batch_x = x[indx[start:end]]
                batch_y = y[indx[start:end]]
                yield batch_x, batch_y

            if tail > 0:
                x_buffer = x_buffer[head:]
                y_buffer = y_buffer[head:]


def nvidia_model():
    input_shape = (image_rows, image_columns, image_channels)
    model = Sequential()
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


def nvidia_model_modified():
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


# if __name__ == '__main__':
#     data = pd.read_csv(os.path.join(DATASET, "driving_log.csv"))
#
#     train_genereator = data_generator(data, GEN_MODE_ALL, preprocess_input=preproccess, batch_size=BATCH_SIZE)
#
#     x,y = next(train_genereator)
#     plt.figure(figsize=(32, 8))
#     a = x[0].shape
#     plt.imshow(x[0])
#     plt.show()
#
#     print("End.")


if __name__ == '__main__':

    data = pd.read_csv(os.path.join(DATASET, "driving_log.csv"))

    train_genereator = data_generator(data, GEN_MODE_ALL, preprocess_input=preproccess, batch_size=BATCH_SIZE)

    model = nvidia_model()
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')

    nb_train = len(data)
    samples_adj = BATCH_SIZE - (nb_train % BATCH_SIZE) if nb_train % BATCH_SIZE > 0 else 0
    samples_per_epoch = nb_train + samples_adj
    model.fit_generator(train_genereator, samples_per_epoch=samples_per_epoch, nb_epoch=EPOCHS, verbose=1,
                        )#callbacks=callbacks)

    print("Saving model weights and configuration file.")

    model.save_weights("model.h5", True)
    with open("model.json", 'w') as outfile:
        outfile.write(model.to_json())
