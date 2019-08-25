import os
from makepairs import get_pair_list

from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,
                                                  load_img)


height = 64
width = 64
dimension = (width, height)

pairs_non_pairs = get_pair_list()

