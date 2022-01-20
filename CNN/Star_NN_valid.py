import os
from os import path
import time
from datetime import date 
import sys
import numpy as np
import matplotlib.pyplot as pyl
import pickle
import heapq

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels

from astropy.visualization import interval, ZScaleInterval

withheld_img = [219580, 219582, 219584, 219586, 219588]
validation_size = 100

# earlier, we set to train on selection from (219590,219620)
# here we validate on 219580 (and can also look at up to 88)

# load specific image (all CCDs) cutouts
files_counted = 0
try:
    for filename in os.listdir(file_dir+ '/NN_data_metadata_111'):
        if filename.endswith("metadata_cutoutData.pickle"):
            #print(files_counted, size_of_data)
            if files_counted >= size_of_data:
                raise BreakException
            print(files_counted, size_of_data)
            #print('file being processed: ', filename)

            with open(file_dir + '/NN_data_metadata_111/' + filename, 'rb') as f:
                [n, cutout, label, y, x, fwhm, inputFile] = pickle.load(f)

            if len(cutout) > 0:
                if cutout.shape == (111,111):
                    if label == 1:
                        good_x_lst.append(x)
                        good_y_lst.append(y)
                        good_fwhm_lst.append(fwhm)
                        good_inputFile_lst.append(inputFile)
                        good_cutouts.append(cutout)
                        files_counted += 1
                    elif label == 0:
                        bad_x_lst.append(x)
                        bad_y_lst.append(y)
                        bad_fwhm_lst.append(fwhm)
                        bad_inputFile_lst.append(inputFile)
                        bad_cutouts.append(cutout)
                        #files_counted += 1

# load model


# show stats analsys


# compare psfs