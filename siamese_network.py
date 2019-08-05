import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Lambda,
                                     MaxPooling2D)
from tensorflow.keras.regularizers import l2
from tensorflow.math import abs, subtract


def fetch_training_data():
    data_loc = os.path.join(os.getcwd(), 'data', 'train_test_images.npz')
    data = np.load(data_loc)
    x_train, y_train = data['x_train'], data['y_train']

    return x_train, y_train


def siamese_model(input_shape):
    """Defines architecture for siamese network. Should be two identically
    weighted conv nets with a distance function layer at the end to differentiate 
    and produce a similarity score. This is similarity based metric not a classification 
    task.
    """
    POOL_SIZE = (2, 2)
    KERNEL_SIZE = (3, 3)
    INPUT_SHAPE = input_shape  # not correct

    # declare tensors for two input images to be compared
    left_input = Input(INPUT_SHAPE)
    right_input = Input(INPUT_SHAPE)

    print(left_input.shape)


    # for filters early layers should have less (more abstract) while later layers gain more filters
    model = Sequential()
    model.add(Conv2D(64, KERNEL_SIZE, activation='relu', input_shape= INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Conv2D(128, KERNEL_SIZE, activation='relu', input_shape= INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Conv2D(256, KERNEL_SIZE, activation='relu', input_shape= INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Flatten())
    model.add(Dense(4056, activation='sigmoid'))

    left_image_encoded = model(left_input)
    right_image_encoded = model(right_input)

    contrastive_layer = Lambda(lambda tensors: abs(subtract(tensors[0], tensors[1])))
    distance = contrastive_layer([left_image_encoded, right_image_encoded])

    prediction = Dense(1, activation='sigmoid')(distance)

    return Model(inputs=[left_input, right_input], outputs=prediction)


def fit_model():
    x_train, y_train = fetch_training_data()
    input_shape = x_train[0][0].shape
    model = siamese_model(input_shape)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(0.0001),
                  metrics=['accuracy'])

    model.fit([x_train[:, 0], x_train[:, 1]], y_train, epochs=10, batch_size=64, validation_split=0.05)


#fetch_training_data()
fit_model()
