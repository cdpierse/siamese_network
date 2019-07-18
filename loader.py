import os
import sys
import itertools
import random 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,
                                                  load_img)

height = 128
width = 128
dimension = (width, height)

def load_images():
    _train_img_list = []
    _test_img_list = []
    img_dir = os.path.join(os.getcwd(),'data') 

    for folder in os.listdir(img_dir):
        if folder == 'train':
            folder_dir = os.path.join(os.getcwd(),'data',folder,'elton_john')
            for file in os.listdir(folder_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = load_img(os.path.join(folder_dir,file),
                                            target_size = dimension,
                                            color_mode ='grayscale')
                    _train_img_list.append(img_to_array(img))
  
        elif folder == 'val':
            folder_dir = os.path.join(os.getcwd(),'data',folder,'elton_john')
            for file in os.listdir(folder_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = load_img(os.path.join(folder_dir,file),
                                        target_size = dimension,
                                        color_mode ='grayscale')
                    _test_img_list.append(img_to_array(img))

    train_imgs = np.array(_train_img_list,dtype ='float32')
    test_imgs = np.array(_test_img_list,dtype='float32')

    return train_imgs,test_imgs

def normalize(img_array):
    return np.divide(img_array,255.0)

if __name__ == '__main__':
    pass


"""TODO - have to build method to   1) get a random sample of data from the df, 
                                    2) split the data into two df's of pairs and not pairs
                                    3) create test and validation set for network based off of these pairs 

"""
