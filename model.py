import os
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Conv2D, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import shuffle

import tensorflow as tf
tf.python.control_flow_ops = tf

epochs = 50
batch_size = 128
dataset_dir = "./dataset"
image_columns = 32
image_rows = 16
image_channels = 1
side_shift = 0.3


def preproccess_image(image):
    image = (cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[:, :, 1]
    image = image.reshape(160, 320, 1)
    image = cv2.resize(image, (image_columns, image_rows))
    return image


def prepare(data):
    x, y = [], []

    for i in range(len(data)):
        line_data = data.iloc[i]
        y_steer = line_data['steering']
        path_center = line_data['center'].strip()
        path_left = line_data['left'].strip()
        path_right = line_data['right'].strip()

        for path, shift in [(path_center, 0), (path_left, side_shift), (path_right, -side_shift)]:
            # read image
            image_path = os.path.join(dataset_dir, path)
            image = cv2.imread(image_path)

            # preprocess image
            image = preproccess_image(image)

            # add image
            x.append(image)
            y.append(y_steer + shift)

            # add flipped image
            image = image[:, ::-1]
            x.append(image)
            y.append(-(y_steer + shift))

    return np.array(x).astype('float32'), np.array(y).astype('float32')


def model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(image_rows, image_columns, image_channels)))
    model.add(Conv2D(2, 3, 3, border_mode='valid', activation='elu'))
    model.add(MaxPooling2D((4, 4), (4, 4), 'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    return model


if __name__ == '__main__':
    print("Loading images...")

    data = pd.read_csv(os.path.join(dataset_dir, "driving_log.csv"))

    X_train, y_train = prepare(data)
    X_train, y_train = shuffle(X_train, y_train)
    X_train = np.expand_dims(X_train, axis=3)

    model = model()
    model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
    checkpoint = ModelCheckpoint("model.h5", monitor='val_mean_squared_error', verbose=1,
                                  save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4,
                                verbose=1, mode='min')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                      callbacks=[checkpoint, early_stop], validation_split=0.15, shuffle=True)

    print("Saving model...")
    with open("model.json", 'w') as outfile:
        outfile.write(model.to_json())

    print("Finished.")