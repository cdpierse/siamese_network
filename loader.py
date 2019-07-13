import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow import keras
import numpy as np
import pandas as pd


height = 128
width = 128
dimension = (width, height)

def read_identities():
    filedir = os.path.join(os.getcwd(),'data','celeba-dataset')
    return pd.read_csv(os.path.join(filedir,'identity_CelebA.txt'),delimiter=' ', names=['Filename','identity_no'])
    
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
   



# X,y = load_images()
# X,y = normalize(X), normalize(y)
# print(X.shape)
# array_to_img(X[0]).show()
print(read_identities().head(100))