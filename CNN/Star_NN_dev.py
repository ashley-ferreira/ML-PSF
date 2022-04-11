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

from astropy.visualization import interval, ZScaleInterval
zscale = ZScaleInterval()

from optparse import OptionParser
parser = OptionParser()

## initializing random seeds for reproducability
# tf.random.set_seed(1234)
# keras.utils.set_random_seed(1234)
np.random.seed(432)

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

model_dir_name_default = pwd + 'Saved_Model/' + \
                datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + '/'
parser.add_option('-m', '--model_dir_name', dest='model_dir_name', \
        default=model_dir_name_default, type='str', \
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

    os.mkdir(options.model_dir_name)
    
    return options.balanced_data_method, options.data_load, options.size_of_data, \
            options.num_epochs, options.model_dir_name, options.cutout_size,  \
            options.pwd, options.training_subdir, options.validation_fraction


def save_scratch_data(size_of_data, cutout_size, model_dir_name, data_dir, balanced_data_method, validation_fraction):
    '''
    Create presaved data file to use for neural network training

    Parameters:    

        size_of_data (int): number of files to acc

        cutout_size (int): defines shape of cutouts (cutout_size, cutout_size)

        model_dir_name (str): directory to save loaded data

        data_dir (str): directory where training data is stored 
        
        balanced_data_method (str): method to balance class weigths 

        validation_fraction (float): fraction of data to save for validation step only

    Returns:
        
        None

    '''

    good_cutouts = [] # label 1
    bad_cutouts = [] # label 0
    good_fwhm_lst = []
    good_x_lst = []
    good_y_lst = []
    good_inputFile_lst = []
    bad_fwhm_lst = []
    bad_x_lst = []
    bad_y_lst = []
    bad_inputFile_lst = []

    files_counted = 0
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('_cutoutData.pickle') and os.path.getsize(data_dir + filename) > 0:
                if files_counted >= size_of_data:
                    break
                print(files_counted, 'out of max size', size_of_data, 'files processed')
                print('file being processed: ', filename)

                with open(data_dir + filename, 'rb') as f:
                    [n, cutout, label, y, x, fwhm, inputFile] = pickle.load(f)

                    if cutout.shape == (cutout_size, cutout_size):
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
                        else:
                            print('ERROR: label is not 1 or 0, excluding cutout')
                            err_log = open(model_dir_name + 'error_log.txt', 'a')
                            err_log.write('Star_NN_dev.py' + filename + 'ERROR: label is not 1 or 0, excluding cutout. label=' + str(label))
                            err_log.close() 
                    else:
                        print('ERROR: wrong cutout shape, excluding cutout')
                        err_log = open(model_dir_name + 'error_log.txt', 'a')
                        err_log.write('Star_NN_dev.py' + filename + 'ERROR: wrong cutout shape. shape=' + str(cutout.shape))
                        err_log.close() 

    except Exception as Argument:
        print('Star_NN_dev.py' + str(Argument))

        # creating/opening a file
        err_log = open(model_dir_name + 'error_log.txt', 'a')

        # writing in the file
        err_log.write('Star_NN_dev.py' + str(Argument))
        
        # closing the file
        err_log.close()    

    # make sure there are more good stars then bad ones
    if len(good_cutouts)>len(bad_cutouts):
        print('ERROR: MORE GOOD STARS THAN BAD STARS')

    # keep all good cutouts
    num_good_cutouts = len(good_cutouts)
    good_x_arr = np.array(good_x_lst)
    good_y_arr = np.array(good_y_lst)
    good_fwhm_arr = np.array(good_fwhm_lst)
    good_inputFile_arr = np.array(good_inputFile_lst)
    bad_x_arr = np.array(bad_x_lst)
    bad_y_arr = np.array(bad_y_lst)
    bad_fwhm_arr = np.array(bad_fwhm_lst)
    bad_inputFile_arr = np.array(bad_inputFile_lst)

    good_cutouts = np.array(good_cutouts)
    good_cutouts = np.expand_dims(good_cutouts, axis=3)

    # add label 1
    label_good = np.ones(num_good_cutouts)
    bad_cutouts = np.array(bad_cutouts, dtype=object) 

    if balanced_data_method == 'even':
        number_of_rows = bad_cutouts.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_good_cutouts, replace=False)
        random_bad_cutouts = bad_cutouts[random_indices, :]
        bad_cutouts = np.expand_dims(random_bad_cutouts, axis=3)
        
        random_bad_x_arr = bad_x_arr[random_indices]
        random_bad_y_arr = bad_y_arr[random_indices]
        random_bad_fwhm_arr = bad_fwhm_arr[random_indices]
        random_bad_inputFile_arr = bad_inputFile_arr[random_indices]

        # add label 0
        label_bad = np.zeros(num_good_cutouts)

    elif balanced_data_method == 'weight':
        #class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        #class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_binary), y_train_binary)
        #class_weights = {1: len(bad_cutouts)/len(good_cutouts), 0: 1.} 
        neg = len(bad_cutouts)
        pos = len(good_cutouts)
        total = pos + neg
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

    # combine arrays 
    cutouts = np.concatenate((good_cutouts, bad_cutouts))
    fwhms = np.concatenate((good_fwhm_arr, random_bad_fwhm_arr))
    files = np.concatenate((good_inputFile_arr, random_bad_fwhm_arr))
    xs = np.concatenate((good_x_arr, random_bad_x_arr))
    ys = np.concatenate((good_y_arr, random_bad_y_arr))

    # make label array for all
    labels = np.concatenate((label_good, label_bad))
    print(str(len(cutouts)) + ' files used')

    skf_v = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction)
    skf_v.split(cutouts, labels)

    for used_index, withheld_index in skf_v.split(cutouts, labels): 
        used_cutouts, withheld_cutouts = cutouts[used_index], cutouts[withheld_index]
        used_labels, withheld_labels = labels[used_index], labels[withheld_index]
        used_xs, withheld_xs = xs[used_index], xs[withheld_index]
        used_ys, withheld_ys = ys[used_index], ys[withheld_index]
        used_files, withheld_files = files[used_index], files[withheld_index]
        used_fwhms, withheld_fwhms = fwhms[used_index], fwhms[withheld_index]

    with open(model_dir_name + '/USED_' + str(cutout_size) + '_presaved_data.pickle', 'wb+') as han:
        pickle.dump([used_cutouts, used_labels, used_xs, used_ys, used_fwhms, used_files], han)

    with open(model_dir_name + '/WITHHELD_' + str(cutout_size) + '_presaved_data.pickle', 'wb+') as han:
        pickle.dump([withheld_cutouts, withheld_labels, withheld_xs, withheld_ys, withheld_fwhms, withheld_files], han)


def load_presaved_data(cutout_size, model_dir_name):
    '''
    Create presaved data file to use for neural network training

    Parameters:    

        cutout_size (int): defines shape of cutouts (cutout_size, cutout_size)

        model_dir_name (str): directory to load data and save regularization params

    Returns:
        
        cutouts (arr): 3D array conisting of 2D image data for each cutout

        labels (arr): 1D array containing 0 or 1 label for bad or good star respectively

    '''
    with open(model_dir_name + 'USED_' + str(cutout_size) + '_presaved_data.pickle', 'rb') as han:
        [cutouts, labels, xs, ys, fwhm, files] = pickle.load(han) 

    cutouts = np.asarray(cutouts).astype('float32')
    std = np.nanstd(cutouts)
    mean = np.nanmean(cutouts)
    cutouts -= mean
    cutouts /= std
    w_bad = np.where(np.isnan(cutouts))
    cutouts[w_bad] = 0.0

    with open(model_dir_name + 'regularization_data.pickle', 'wb+') as han:
        pickle.dump([std, mean], han)

    return [cutouts, labels, xs, ys, fwhm, files]


def train_CNN(model_dir_name, num_epochs, data):
    '''
    Sets up and trains Convolutional Neural Network (CNN).
    Plots accuracy and loss over each training epoch.

    Parameters:    

        model_dir_name (str): directory to store model
        
        num_epochs (int): number of epochs to train for

        cutouts (arr): 3D array conisting of 2D image data for each cutout

        labels (arr): 1D array containing 0 or 1 label for bad or good star respectively

        xs (arr): 1D array containing central x position of cutout 

        ys (arr): 1D array containing central y position of cutout 

        fwhms (arr): 1D array containing fwhm values for each cutout 
        
        files (arr): 1D array containing file names for each cutout

    Return: 

        X_train (arr): X values (images) for training
        
        y_train (arr): real y values (labels) for training
        
        X_test (arr): X values (images) for testing 
        
        y_test (arr): real y values (labels) for testing 

    '''
    # unpack presaved data
    cutouts, labels, xs, ys, fwhms, files = data[0], data[1], data[2], data[3], data[4], data[5]

    # section for setting up some flags and hyperparameters
    batch_size = 16 
    dropout_rate = 0.2
    test_fraction = 0.05 
    learning_rate = 0.001

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
    
    # unique_labs = len(np.unique(y_train))
    unique_labels = 2
    y_train_binary = keras.utils.np_utils.to_categorical(y_train, unique_labels)

    # train the model
    cn_model = convnet_model(X_train.shape[1:], unique_labs=unique_labels, dropout_rate=dropout_rate)
    cn_model.summary()

    opt = Adam(learning_rate=learning_rate) 
    cn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

    start = time.time()
    X_train = np.asarray(X_train).astype('float32')
    y_train_binary = np.asarray(y_train_binary).astype('float32')

    classifier = cn_model.fit(X_train, y_train_binary, epochs=num_epochs, batch_size=batch_size)

    end = time.time()
    print('Process completed in', round(end-start, 2), ' seconds')

    # save trained model 
    cn_model.save(model_dir_name + 'model_' + str(end))

    # plot accuracy/loss versus epoch
    fig1 = pyl.figure(figsize=(10,3))

    ax1 = pyl.subplot(121)
    ax1.plot(classifier.history['accuracy'], color='darkslategray', linewidth=2)
    #ax1.plot(classifier.history['val_accuracy'], color='blue', linewidth=2)
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')

    ax2 = pyl.subplot(122)
    ax2.plot(classifier.history['loss'], color='crimson', linewidth=2)
    #ax2.plot(classifier.history['val_loss'], color='blue', linewidth=2)
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')

    fig1.savefig(model_dir_name +'/plots/'+'NN_training_history' + str(end) + '.png')

    pyl.show()
    pyl.close()
    pyl.clf()

    return cn_model, X_train, y_train, X_test, y_test

def test_CNN(cn_model, model_dir_name, X_train, y_train, X_test, y_test):
    ''' 
    Tests previously trained Convolutional Neural Network (CNN).
    Plots confusion matrix for 50% confidence cutoff.

    Parameters:    

        X_train (arr): X values (images) for training
        
        y_train (arr): real y values (labels) for training
        
        X_test (arr): X values (images) for testing 
        
        y_test (arr): real y values (labels) for testing y

    Return: 

        None

    '''
    # get the model output classifications for the train and test sets
    X_test = np.asarray(X_test).astype('float32')
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
    pyl.title('Confusion matrix')
    pyl.colorbar()
    pyl.xlabel('Predicted labels')
    pyl.ylabel('True labels')
    pyl.show()
    fig2.savefig(model_dir_name +'/plots/'+'NN_confusion_matrix.png')
    pyl.close()

def main():

    balanced_data_method, data_load, size_of_data, num_epochs, \
    model_dir_name, cutout_size, pwd, training_subdir, validation_fraction = get_user_input()

    data_dir = pwd + training_subdir

    if data_load == 'scratch':
        save_scratch_data(size_of_data, cutout_size, model_dir_name, data_dir, balanced_data_method, validation_fraction)

    cn_model, X_train, y_train, X_test, y_test = train_CNN(model_dir_name, num_epochs, load_presaved_data(cutout_size, model_dir_name))

    test_CNN(cn_model, model_dir_name, X_train, y_train, X_test, y_test)
    
if __name__ == '__main__':
    main()