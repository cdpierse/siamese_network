import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras import Sequential


def fetch_training_data():
    data_loc = os.path.join(os.getcwd(),'data','train_test_images.npz')
    data = np.load(data_loc)
    x_train, y_train = data['x_train'], data['y_train']

def siamese_model():

    model = Sequential()


fetch_training_data()