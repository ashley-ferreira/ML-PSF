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

#from resnet_model_v2 import convnet_model_resnet
from convnet_model_lesslayers import convnet_model_lesslayers

from astropy.visualization import interval, ZScaleInterval
zscale = ZScaleInterval()

from optparse import OptionParser
parser = OptionParser()

from tempfile import TemporaryFile
outfile = TemporaryFile()

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

    return options.balanced_data_method, options.data_load, options.size_of_data, \
            options.num_epochs, model_dir_name, options.cutout_size,  \
            options.pwd, options.training_subdir, options.validation_fraction


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

    return cutouts


def load_presaved_data(cutout_size, model_dir_name):
    '''
    '''
    print('Begin data loading...')
    with open(model_dir_name + 'USED_' + str(cutout_size) + '_presaved_data.pickle', 'rb') as used_c:
        [cutouts, labels, xs, ys, fwhms, files] = pickle.load(used_c) 
    print('Data all loaded')
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
    stds_lst, seconds_lst, stds_n_lst, seconds_n_lst = [], [], [], []
    # need to recalculate this
    '''
    cutout_dir = '/arc/projects/uvickbos/ML-PSF/NN_data_111/'
    for c, f in zip(cutouts, files): 
        print(f)
        # read in saved cutout file created from HSCgetStars_main 
        c_f = str(cutout_dir+f.replace('.fits', '_111_cutouts_savedFits.pickle')) # need it to end like expected
        
        with open(c_f, 'rb') as old_c:
            [stds, seconds, peaks, xs_old, ys_old, cutouts_old, fwhm_old, inputFile_old] = pickle.load(old_c)

        # MAKE LIST OF STDs and second peak, calc flocally or call?
        ## select only those stars with really low STD
        w = np.where(stds/np.std(stds)<0.001)
        stds = stds[w]
        seconds = seconds[w]
        peaks = peaks[w]
        s = np.std(stds)

        ## find the best 25 stars (the closest to the origin in 
        ## weighted STD and second highest pixel value)
        dist = ((stds/s)**2 + (seconds/peaks)**2)**0.5

        stds_lst.append(stds)
        seconds_lst.append(seconds)
        stds_n_lst.append(stds/s)
        seconds_n_lst.append(peaks)
    '''
    # DOING VERY SIMPLIFIED STD BELOW
    # usually done differently and on rem_cutout
    peaks, stds, seconds, ss = [], [], [], []
    print(cutouts.shape)
    c_temp = np.copy(cutouts)
    c_temp = np.squeeze(c_temp, axis=3)
    for cutout in c_temp:
        (aa,bb) = cutout.shape
        peak = np.max(cutout)
        peaks.append(peak)
        s = np.nanstd(cutout)
        ss.append(s)
        sorted = np.sort(cutout.reshape(aa*bb))
        second_highest = sorted[-2]
        seconds.append(second_highest)
        seconds_n_lst.append(second_highest/peak)

            
    # store everything in numpy array format
    std = np.array(stds)
    seconds = np.array(seconds)
    xs = np.array(xs)
    ys = np.array(ys)
    peaks = np.array(peaks)

    cutouts = np.asarray(cutouts).astype('float32')
    std = np.nanstd(cutouts)
    mean = np.nanmean(cutouts)
    cutouts = regularize(cutouts, mean, std)
    print('Data all regulatized')
    with open(model_dir_name + 'regularization_data.pickle', 'wb+') as han:
        pickle.dump([std, mean], han)

    return [cutouts, labels, xs, ys, fwhms, files, ss, seconds, ss, seconds_n_lst]


def train_CNN(model_dir_name, num_epochs, data):
    '''
    '''
    
    # unpack presaved data
    cutouts, labels, xs, ys, fwhms, files = data[0], data[1], data[2], data[3], data[4], data[5]
    stds_lst, seconds_lst, stds_n_lst, seconds_n_lst = data[6], data[7], data[8], data[9]

    
    X_train = cutouts
    y_train = labels

    
    unique_labs = len(np.unique(y_train)) # should be 2
    y_train_binary = keras.utils.np_utils.to_categorical(y_train, unique_labs)

    X_train = np.asarray(X_train).astype('float32')
    y_train_binary = np.asarray(y_train_binary).astype('float32')


    model_dir_name_x = model_dir_name + '/models_each_10epochs_BASIC/model_60'
    cn_model = keras.models.load_model(model_dir_name_x)

    # make predictions for colours
    conf = cn_model.predict(X_train)
    good_conf = []
    for c in conf:
        print(c)
        good_conf.append(c[1])

    good_conf = np.array(good_conf)
    print(good_conf.shape)

    # plot accuracy/loss versus epoch
    fig1 = pyl.figure(figsize=(10,3))

    stds_lst, seconds_lst, stds_n_lst, seconds_n_lst

    ax1 = pyl.subplot(121)
    for l in y_train:
        if l==1:
            ax1.loglog(stds_n_lst, seconds_n_lst, 'o', label='label=1', alpha=0.1, color='red') 
        elif l==0:
            ax1.loglog(stds_n_lst, seconds_n_lst, 'o', label='label=0', alpha=0.1, color='blue') 
    #ax1.legend()
    ax1.set_title('Data Labels')
    ax1.set_ylabel('2nd Peaks')
    ax1.set_xlabel('STDs')
    # make sure 50 is white? (check how far off)
    # add color bad for label 1 confidence val? <50 = 0

    ax2 = pyl.subplot(122)
    cb = ax2.scatter(stds_n_lst, seconds_n_lst, 'o', c=good_conf, alpha=0.1, cmap='bwr')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.set_title('Model Predictions')
    ax2.set_ylabel('2nd Peaks')
    ax2.set_xlabel('STDs')
    ax2.set_clim(0,1)
    # colorscale
    clb = fig1.colorbar(cb, cax=ax2, orientation='vertical')
    clb.ax2.set_title('good star confidence')
    #cb.set_label('Color Scale')

    fig1.savefig(model_dir_name +'/plots/'+'scatter_n.png')

    pyl.show()
    pyl.close()
    pyl.clf()

def main():

    balanced_data_method, data_load, size_of_data, num_epochs, \
    model_dir_name, cutout_size, pwd, training_subdir, validation_fraction = get_user_input()

    data_dir = pwd + training_subdir

    train_CNN(model_dir_name, num_epochs, load_presaved_data(cutout_size, model_dir_name))
    
if __name__ == '__main__':
    main()