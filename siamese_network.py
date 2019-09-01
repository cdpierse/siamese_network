import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Lambda,
                                     MaxPooling2D, Dropout, SpatialDropout2D)
from tensorflow.keras.regularizers import l2
from keras.backend import abs
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score


def fetch_train_test():
    data_loc = os.path.join(os.getcwd(), 'data', 'train_test_images.npz')
    data = np.load(data_loc)
    return data


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

    model.add(Conv2D(32, KERNEL_SIZE, activation='relu',
                     input_shape=INPUT_SHAPE, kernel_regularizer=l2(0.4)))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Conv2D(64, KERNEL_SIZE, activation='relu',
                     kernel_regularizer=l2(0.3)))
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Conv2D(128, KERNEL_SIZE, activation='relu',
                     kernel_regularizer=l2(0.01)))

    model.add(Flatten())
    model.add(Dropout(0.45))
    model.add(Dense(2056, activation='sigmoid', kernel_regularizer=l2(0.001)))

    left_image_encoded = model(left_input)
    right_image_encoded = model(right_input)

    contrastive_layer = Lambda(lambda tensors: abs(tensors[0] - tensors[1]))
    distance = contrastive_layer([left_image_encoded, right_image_encoded])

    prediction = Dense(1, activation='sigmoid')(distance)

    return Model(inputs=[left_input, right_input], outputs=prediction)


def fit_model(lr=None, batch_size=None, val_split=None, epochs=None):
    if lr is None:
        optimizer = tf.keras.optimizers.Adam(0.0001)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    if batch_size is None:
        batch_size = 32

    if val_split is None:
        val_split = 0.15

    if epochs is None:
        epochs = 100

    train_test_data = fetch_train_test()
    x_train, y_train = train_test_data['x_train'], train_test_data['y_train']
    input_shape = x_train[0][0].shape
    model = siamese_model(input_shape)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit([x_train[:, 0], x_train[:, 1]], y_train,
              epochs=epochs, batch_size=batch_size, validation_split=val_split)

    x_test, y_test = train_test_data['x_test'], train_test_data['y_test']

    print(model.evaluate([x_test[:, 0], x_test[:, 1]], y_test))
    print(model.metrics_names)
    model.save('model.h5')

fit_model()
