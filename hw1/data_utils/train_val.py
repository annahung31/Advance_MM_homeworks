# This module parses collected and refined dataset and divides it into
# Train, Validation and Test splits. 
# First, we have to read the data
# Second, we have to place it into numpy array 
import os, sys
import pathlib
import argparse
import numpy as np
import pandas as pd
# from subprocess import check_output
#print(check_output(["ls", "../input"]))
from PIL import Image
from time import time
from time import sleep
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import random
import math
import shutil

import ipdb


# arguments to pass in command line 
parser = argparse.ArgumentParser(description='Rename images in the folder according to LFW format: Name_Surname_0001.jpg, Name_Surname_0002.jpg, etc.')
parser.add_argument('--dataset-dir', default='', help='Full path to the directory with peeople and their names, folder should denote the Name_Surname of the person')
#parser.add_argument('--target-dir', default='', help='Full path to the directory where our identified images should be saved.')

# reading the passed arguments
args = parser.parse_args()
data_dir = args.dataset_dir
tr_dir = '../train/tr_split'
val_dir = '../train/val_split'



cont = 0

for folder in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, folder)):
        continue
    i = 1
    # print(folder)
    fold = data_dir + '/' + folder
    cnt = 0     # count the number of files in a folder
    onlyfiles = []
    for img in os.listdir(fold):
        # print(img)
        
        if os.path.splitext(img)[1] == '.jpg':
            onlyfiles.append(img)
            cnt += 1
    print(cnt)
    percent = math.floor((cnt * 90) / 100)
    
    print(percent)

    train_folder = tr_dir + '/' + folder
    val_folder = val_dir + '/' + folder

    # create folder for current person's test images
    
    
    os.makedirs(train_folder, exist_ok = True)
    os.makedirs(val_folder, exist_ok = True)
    

    idx = [i for i in range(cnt)]
    random.seed(100)
    tr_idx = random.sample(idx, percent)
    for i, f in enumerate(onlyfiles):
        
        file_to_move = fold + '/' + f
        if i in tr_idx:
            new_file = train_folder + '/' + f
        else:
            new_file = val_folder + '/' + f

        shutil.copyfile(file_to_move, new_file)
        cont += 1

print(cont)
