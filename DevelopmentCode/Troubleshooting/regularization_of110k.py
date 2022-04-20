import os
from os import path
import time
from datetime import date 
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as pyl
import pickle
import heapq

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import csv


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
zscale = ZScaleInterval()

from optparse import OptionParser
parser = OptionParser()

## initializing random seeds for reproducability
# tf.random.set_seed(1234)
# keras.utils.set_random_seed(1234)
np.random.seed(123) # cahnged from 432

pwd = '/arc/projects/uvickbos/ML-PSF/'
parser.add_option('-p', '--pwd', dest='pwd', 
        default=pwd, type='str', 
        help=', default=%default.')

# likely removing this option as it hasnt worked well and right now not an option for training
parser.add_option('-b', '--balanced_data_method', dest='balanced_data_method', 
        default='even', type='str', 
        help='method to balanced classes (even or weighted), default=%default.')

parser.add_option('-d', '--data_load', dest='data_load', 
        default='scratch', type='str', 
        help='how to load data (presaved or scratch), default=%default.')

parser.add_option('-s', '--size_of_data', dest='size_of_data', 
        default='1000', type='int', 
        help='number of cutouts to use, default=%default.')

parser.add_option('-n', '--num_epochs', dest='num_epochs', 
        default='500', type='int', 
        help='how many epochs to train for, default=%default.')

model_dir = pwd + 'Saved_Model/' 
model_name_default = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + '/'
parser.add_option('-m', '--model_dir_name', dest='model_name', \
        default=model_name_default, type='str', \
        help='name for model directory, default=%default.')

cutout_size = 111
parser.add_option('-c', '--cutout_size', dest='cutout_size', \
        default=cutout_size, type='int', \
        help='c is size of cutout required, produces (c,c) shape, default=%default.')

parser.add_option('-t', '--training_subdir', dest='training_subdir', \
        default='NN_data_' + str(cutout_size) + '/', type='str', \
        help='subdir in pwd for training data, default=%default.')

parser.add_option('-v', '--validation_fraction', dest='validation_fraction', \
        default='0.1', type='float', \
        help='fraction of images saved to only use in validation step, default=%default.')

def get_user_input():
    '''
    Gets user user preferences for neural network training parameters/options

    Parameters:    

        None

    Returns:
        
        balanced_data_method (str): even or weighted classes
        
        data_load (str): using presaved data set or preparing from scratch

        size_of_data (int): size of data to load from scratch, 0 if using presaved
        
        num_epochs (int): number of epochs to train neural network for

        model_dir_name (str): directory to store all outputs

        cutout_size (int): is size of cutout required, produces (cutout_size,cutout_size) shape

        pwd (str): working directory, will load data from subdir and save model into subdir

        training_sub_dir (str): subdir in pwd for training data

    '''
    (options, args) = parser.parse_args()

    # can't get exist_ok=True option working so this is solution
    model_dir_name = model_dir + options.model_name
    if not(os.path.exists(model_dir_name)):
        os.mkdir(model_dir_name)
    plots_dir = model_dir_name + 'plots/'
    if not(os.path.exists(plots_dir)):
        os.mkdir(plots_dir)
    submodels_dir = model_dir_name + 'models_each_10epochs/'
    if not(os.path.exists(submodels_dir)):
        os.mkdir(submodels_dir)
    
    return options.balanced_data_method, options.data_load, options.size_of_data, \
            options.num_epochs, model_dir_name, options.cutout_size,  \
            options.pwd, options.training_subdir, options.validation_fraction


def load_presaved_data(cutout_size, model_dir_name):
    '''
    Create presaved data file to use for neural network training

    Parameters:    

        cutout_size (int): defines shape of cutouts (cutout_size, cutout_size)

        model_dir_name (str): directory to load data and save regularization params

    Returns:
        
        data (lst), which consists of:

            cutouts (arr): 3D array conisting of 2D image data for each cutout

            labels (arr): 1D array containing 0 or 1 label for bad or good star respectively

            xs (arr): 1D array containing central x position of cutout 

            ys (arr): 1D array containing central y position of cutout 

            fwhms (arr): 1D array containing fwhm values for each cutout 
            
            files (arr): 1D array containing file names for each cutout

    '''
    with open(model_dir_name + 'USED_' + str(cutout_size) + '_presaved_data.pickle', 'rb') as han:
        [cutouts, labels, xs, ys, fwhm, files] = pickle.load(han) 

    indiv_std = []
    indiv_median = []
    indiv_maxpix = []
    indiv_minpix = []
    indiv_files = []
    x_lst = []
    y_lst = []

    f = open(model_dir_name + 'indiv_reg_data.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['std','mean','pix_max','pix_min','file','x','y'])
    for i in range(len(cutouts)): #only one cutout?
        cutout = np.asarray(cutouts[i]).astype('float32')
        std = np.nanstd(cutout)
        mean = np.nanmean(cutout)
        pix_max = cutout.max()
        pix_min = cutout.min()
        indiv_file = files[i]
        indiv_std.append(std)
        indiv_median.append(mean)
        indiv_maxpix.append(pix_max)
        indiv_minpix.append(pix_min)
        indiv_files.append(indiv_file)
        x_lst.append(xs[i])
        y_lst.append(ys[i])
        writer.writerow([std,mean,pix_max,pix_min,indiv_file,xs[i],ys[i]])

    f.close()

    with open(model_dir_name + 'indiv_reg_data.pickle', 'wb+') as han:
        pickle.dump([indiv_std,indiv_median,indiv_maxpix,indiv_minpix,indiv_files], han)

    return [cutouts, labels, xs, ys, fwhm, files]

def main():

    balanced_data_method, data_load, size_of_data, num_epochs, \
    model_dir_name, cutout_size, pwd, training_subdir, validation_fraction = get_user_input()

    data_dir = pwd + training_subdir

    load_presaved_data(cutout_size, model_dir_name)
    
if __name__ == '__main__':
    main()