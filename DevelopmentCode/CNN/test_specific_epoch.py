import os
import sys
import numpy as np
import matplotlib.pyplot as pyl
import pickle

import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib as mpl

from astropy.visualization import interval, ZScaleInterval
zscale = ZScaleInterval()

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

from convnet_model import convnet_model
from convnet_model_complex import convnet_model_complex

from astropy.visualization import interval, ZScaleInterval
zscale = ZScaleInterval()

np.random.seed(432)

pwd = '/arc/projects/uvickbos/ML-PSF/'
model_dir_name = pwd+'Saved_Model/2022-04-23-13:53:44/'
test_fraction = 0.05


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
        [cutouts, labels, xs, ys, fwhms, files] = pickle.load(han) 

    # temporary add for old 110k data:
    '''
    for i in range(len(cutouts)):
        cutout = np.asarray(cutouts[i]).astype('float32')
        if cutout.min() < -2000 or cutout.max() > 130000:
            cutouts = np.delete(cutouts,i)
            labels = np.delete(labels,i)
            xs = np.delete(xs,i)
            ys = np.delete(ys,i) 
            fwhms = np.delete(fwhms,i)
            files = np.delete(files,i)
        else:
            if cutouts.min() < -200 or cutout.max() > 65536:
                labels[i] = 0
    '''

    cutouts = np.asarray(cutouts).astype('float32')
    std = np.nanstd(cutouts)
    mean = np.nanmean(cutouts)
    cutouts = regularize(cutouts, mean, std)

    with open(model_dir_name + 'regularization_data.pickle', 'wb+') as han:
        pickle.dump([std, mean], han)

    return [cutouts, labels, xs, ys, fwhms, files]


def regularize(cutouts, mean, std):
    '''
    Regularizes either single cutout or array of cutouts

    Parameters:

        cutouts (arr): cutouts to be regularized

        mean (float): mean used in training data regularization  

        std (float): std used in training data regularization

    Returns:

        regularized_cutout (arr): regularized cutout
    
    '''
    cutouts = np.asarray(cutouts).astype('float32')
    cutouts -= mean
    cutouts /= std
    w_bad = np.where(np.isnan(cutouts))
    cutouts[w_bad] = 0.0
    regularized_cutout = cutouts

    return regularized_cutout

def train_CNN(data):
    '''
    Sets up and trains Convolutional Neural Network (CNN).
    Plots accuracy and loss over each training epoch.

    Parameters:    

        model_dir_name (str): directory to store model
        
        num_epochs (int): number of epochs to train for

        data (lst), which consists of:

            cutouts (arr): 3D array conisting of 2D image data for each cutout

            labels (arr): 1D array containing 0 or 1 label for bad or good star respectively

            xs (arr): 1D array containing central x position of cutout 

            ys (arr): 1D array containing central y position of cutout 

            fwhms (arr): 1D array containing fwhm values for each cutout 
            
            files (arr): 1D array containing file names for each cutout

    Return: 

        cn_model (keras model): trained neural network

        X_train (arr): X values (images) for training
        
        y_train (arr): real y values (labels) for training
        
        X_test (arr): X values (images) for testing 
        
        y_test (arr): real y values (labels) for testing 

    '''

    # unpack presaved data
    cutouts, labels, xs, ys, fwhms, files = data[0], data[1], data[2], data[3], data[4], data[5]

    ### now divide the cutouts array into training and testing datasets.
    skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction)
    print(skf)
    skf.split(cutouts, labels)

    for train_index, test_index in skf.split(cutouts, labels):
        X_train, X_test = cutouts[train_index], cutouts[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        xs_train, xs_test = xs[train_index], xs[test_index]
        ys_train, ys_test = xs[train_index], xs[test_index]
        files_train, files_test = files[train_index], files[test_index]
        fwhms_train, fwhms_test = fwhms[train_index], fwhms[test_index]

    unique_labs = len(np.unique(y_train)) # should be 2
    y_train_binary = keras.utils.np_utils.to_categorical(y_train, unique_labs)
   
    X_train = np.asarray(X_train).astype('float32')
    y_train_binary = np.asarray(y_train_binary).astype('float32')

    cn_model = keras.models.load_model(pwd+ 'Saved_Model/2022-04-23-13:53:44/'+ 'models_each_10epochs/' + "model_350")
    
    return cn_model, X_train, y_train, X_test, y_test

def test_CNN(cn_model, model_dir_name, X_train, y_train, X_test, y_test):
    ''' 
    Tests previously trained Convolutional Neural Network (CNN).
    Plots confusion matrix for 50% confidence cutoff.

    Parameters:    

        cn_model (keras model): trained neural network

        X_train (arr): X values (images) for training
        
        y_train (arr): real y values (labels) for training
        
        X_test (arr): X values (images) for testing 
        
        y_test (arr): real y values (labels) for testing y

    Return: 

        None

    '''

    # get the model output classifications for the train and test sets
    X_test = np.asarray(X_test).astype('float32')
    unique_labs = len(np.unique(y_test)) # should be 2
    y_test_binary = keras.utils.np_utils.to_categorical(y_test, unique_labs)
    y_test_binary = np.asarray(y_test_binary).astype('float32')
    preds_test = cn_model.predict(X_test, verbose=1)
    preds_train = cn_model.predict(X_train, verbose=1)

    # evanluate test set (50% confidence threshold)
    results = cn_model.evaluate(X_test, y_test_binary)
    print("test loss, test acc:", results)

    # plot confusion matrix (50% confidence threshold)
    fig2 = pyl.figure()
    y_test_binary = np.argmax(y_test_binary, axis = 1) 
    preds_test_binary = np.argmax(preds_test, axis = 1)
    cm = confusion_matrix(y_test_binary, preds_test_binary)
    pyl.matshow(cm)

    for (i, j), z in np.ndenumerate(cm):
        pyl.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    pyl.title('Confusion matrix (testing data)')
    pyl.colorbar()
    pyl.xlabel('Predicted labels')
    pyl.ylabel('True labels')
    pyl.show()
    fig2.savefig(model_dir_name +'plots/'+'NN_confusion_matrix.png')
    pyl.close()
    pyl.clf()


cn_model, X_train, y_train, X_test, y_test = train_CNN(load_presaved_data(111, model_dir_name))

test_CNN(cn_model, model_dir_name, X_train, y_train, X_test, y_test)