import os
import sys
import itertools
import shutil
import random 
import numpy as np
import math
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,
                                                  load_img)
from makepairs import get_pair_list

height = 64
width = 64
dimension = (width, height)

pairs_non_pairs = get_pair_list()

def make_image_sample_folder():
    # this will probably be a slow function that I should run only when a new sample size is needed. 
    source_path = os.path.join(os.getcwd(),'data','celeba-dataset','img_align_celeba') 
    destination = os.path.join(os.getcwd(),'data','faces_aligned')

    if not os.path.exists(destination):
        os.makedirs(destination)

    shutil.rmtree(destination)  # should delete the current contnets of the directory so they can be loaded with new imgs

    copy_files = []

    for i in range(len(pairs_non_pairs)):
        copy_files.append(pairs_non_pairs.iloc[i].file1)
        copy_files.append(pairs_non_pairs.iloc[i].file2)


    for file in os.listdir(source_path):
        if file in copy_files:
            source_file = os.path.join(source_path,file)
            shutil.copy(source_file,destination)

def load_images_to_arrays():
    """ 
    Maps the images in face_aligned folder with the index in the pairs_non_pairs csv/df.
    Loads each image pair/non_pair from the faces_aligned folder and creates a numpy array of tuples where each 
    tuple is a pair of images to bed fed into the network. Corresponding y list is target where 1 means pair and 
    0 means non pair. Saves the train and test array of tuple to file as compressed npz. 
    """
    img_dir = os.path.join(os.getcwd(),'data','faces_aligned') 

    convert_targets_to_binary(pairs_non_pairs)

    img_tuples_list = []
    y = []
    img_added_counter = 0
    img1 = None
    img2 = None

    for i in range(len(pairs_non_pairs)):
        print(f'images added so far is: {img_added_counter}')
        for file in os.listdir(img_dir):
            if file == pairs_non_pairs.iloc[i].file1:
                img1 = load_img(os.path.join(img_dir,file),
                color_mode ='grayscale',
                target_size = dimension)
                
                img1 = img_to_array(img1)
                img1 = normalize(img1)
                

            elif file == pairs_non_pairs.iloc[i].file2:
                img2 = load_img(os.path.join(img_dir,file),
                color_mode ='grayscale',
                target_size= dimension)
                img2 = img_to_array(img2)
                img2 = normalize(img2)

            img_tuple = (img1,img2)

        img_tuples_list.append(img_tuple)
        y.append(pairs_non_pairs.iloc[i].match)
        img_added_counter += 1
    
    array_of_tuples = np.asarray(img_tuples_list)

    print(f'length of img tuples is {len(img_tuples_list)}')

    split_index = math.ceil(len(array_of_tuples) * 0.8)

    x_train = array_of_tuples[:split_index]
    x_test = array_of_tuples[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    print(type(x_train))
    
    save_file = os.path.join(os.getcwd(),'data','train_test_images')
    np.savez_compressed(save_file, x_train= x_train, y_train=y_train,
                        x_test= x_test, y_test= y_test)

    print(f'len of x_train is {len(x_train)} and len of x_test is {len(x_test)}')

   
def convert_targets_to_binary(pairs_non_pairs):
    pairs_non_pairs.loc[pairs_non_pairs['match'] == 'Yes', 'match'] = 1
    pairs_non_pairs.loc[pairs_non_pairs['match'] == 'No', 'match'] = 0

    return pairs_non_pairs

def normalize(img_array):
    return np.divide(img_array,255.0)

if __name__ == '__main__':
    # make_image_sample_folder()  #only run when you want a new sample base
    load_images_to_arrays()
