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

from astropy.visualization import interval, ZScaleInterval
zscale = ZScaleInterval()

## initializing random seeds for reproducability
# tf.random.set_seed(1234)
# keras.utils.set_random_seed(1234)
np.random.seed(432)

pwd = '~/PSF_star_selection/'

def get_user_input():
    '''
    Prompts user for neural network training parameters/options

    Parameters:    

        None

    Returns:
        
        balanced_data_method (str): even or weighted classes
        
        data_load (str): using presaved data set or preparing from scratch

        size_of_data (int): size of data to load from scratch, 0 if using presaved
        
        num_epochs (int): number of epochs to train neural network for

        model_dir_name (str): directory to store all outputs

    '''
    val = input("Change default values (Y/N): ")
    if val == 'Y':
        balanced_data_method = input("Method to balanced classes (even or weighted):")
        data_load = input("How to load data (presaved or scratch):")

        if data_load == 'scratch':
            size_of_data = int(input("Number of cutouts to use:"))
        else: 
            size_of_data = 0

        cutout_size = int(input("Size of cutouts (default 111):"))
        num_epochs = int(input("How many epochs to train for:"))
        model_dir_name = intput("Name for model directory")
    else: 
        balanced_data_method = 'even' 
        data_load = 'scratch'
        num_epochs = 500
        cutout_size = 111
        model_dir_name = pwd + 'Saved_Model/' + 
                            datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + '/'
    
    return balanced_data_method, data_load, size_of_data, 
                 num_epochs, cutout_size, model_dir_name


def save_scratch_data(size_of_data, cutout_size, model_dir_name):
    '''
    Create presaved data file to use for neural network training

    Parameters:    

        size_of_data (int): number of files to acc

        cutout_size (int): defines shape of cutouts (cutout_size, cutout_size)

        model_dir_name (str): directory to save loaded data

    Returns:
        
        None

    '''

    good_cutouts = [] # label 1
    bad_cutouts = [] # label 0
    cutout_len = []
    good_fwhm_lst = []
    good_x_lst = []
    good_y_lst = []
    good_inputFile_lst = []
    bad_fwhm_lst = []
    bad_x_lst = []
    bad_y_lst = []
    bad_inputFile_lst = []

    data_dir = pwd + 'NN_data_' + str(cutout_size) + '/'

    files_counted = 0
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('_cutoutData.pickle'):
                if files_counted >= size_of_data:
                    break
                print(files_counted, size_of_data)
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
                    else:
                        continue
    except Exception as e: 
        print(e)  

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
    print(good_cutouts.shape)
    good_cutouts = np.expand_dims(good_cutouts, axis=3)
    print(good_cutouts.shape)

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


    # combine arrays 
    cutouts = np.concatenate((good_cutouts, bad_cutouts))
    fwhms = np.concatenate((good_fwhm_arr, random_bad_fwhm_arr))
    files = np.concatenate((good_inputFile_arr, random_bad_fwhm_arr))
    xs = np.concatenate((good_x_arr, random_bad_x_arr))
    ys = np.concatenate((good_y_arr, random_bad_y_arr))

    # make label array for all
    labels = np.concatenate((label_good, label_bad))
                
    print(str(len(cutouts)) + ' files used')

    with open(model_dir_name + str(cutout_size) + '_presaved_data.pickle', 'wb+') as han:
        pickle.dump([cutouts, labels, xs, ys, fwhm, files], han)


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
    
    with open(model_dir_name +  str(cutout_size) + '_presaved_data.pickle', 'rb') as han:
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

    return cutouts, labels

def convnet_model(input_shape, training_labels, unique_labs, dropout_rate):
    '''
    Defines the 2D Convolutional Neural Network (CNN)

    Parameters:    

        input_shape (arr): input shape for network

        training_labels (arr): training labels

        unique_labels (int): number unique labels (for good and bad stars = 2)

        dropout_rate (float): dropout rate

    Returns:
        
        model (keras model class): convolutional neural network to train

    '''

    model = Sequential()

    #hidden layer 1
    model.add(Conv2D(filters=16, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    #hidden layer 2 with Pooling
    model.add(Conv2D(filters=16, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    model.add(BatchNormalization())

    #hidden layer 3 with Pooling
    model.add(Conv2D(filters=8, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(unique_labs, activation='softmax')) 
    #model.add(Activation("softmax"))

    return model

def train_CNN():
    '''
    Sets up and trains Convolutional Neural Network (CNN).
    Plots accuracy and loss over each training epoch.

    Parameters:    

        input_shape (arr): input shape for network

        training_labels (arr): training labels

        unique_labels (int): number unique labels (for good and bad stars = 2)

        dropout_rate (float): dropout rate

    Returns:
        
        model (keras model class): convolutional neural network to train

    '''

    # section for setting up some flags and hyperparameters
    batch_size = 16 
    dropout_rate = 0.2
    test_fraction = 0.05 
    learning_rate = 0.001
    max_size = 111 

    ### now divide the cutouts array into training and testing datasets.
    skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction)
    print(skf)
    skf.split(cutouts, labels)

    for train_index, test_index in skf.split(cutouts, labels):
        X_train, X_test = cutouts[train_index], cutouts[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        xs_train, xs_test = xs[train_index], xs[test_index]
        files_train, files_test = files[train_index], files[test_index]
        fwhms_train, fwhms_test = fwhms[train_index], fwhms[test_index]
    
    #unique_labs = len(np.unique(y_train))
    #print(unique_labels)
    y_train_binary = keras.utils.np_utils.to_categorical(y_train, 2) #unique_labels)

    ### train the model!
    cn_model = convnet_model(X_train.shape[1:], training_labels=y_train, unique_labs=2)
    #cn_model = convnet_model(X_train.shape[1:], y_train)
    cn_model.summary()

    opt = Adam(learning_rate=learning_rate) # add lr to top param
    cn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])#, learning_rate=0.1)

    start = time.time()
    X_train = np.asarray(X_train).astype('float32')
    y_train_binary = np.asarray(y_train_binary).astype('float32')

    classifier = cn_model.fit(X_train, y_train_binary, epochs=num_epochs, batch_size=batch_size)

    end = time.time()
    print('Process completed in', round(end-start, 2), ' seconds')

    # save trained model 
    cn_model.save(model_dir_name + 'model_' + str(end))

    # plot accuracy/loss versus epoch
    fig = pyl.figure(figsize=(10,3))

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

    fig.savefig(file_dir+'/NN_plots/'+'NN_scores_plot' + str(end) + '.png')

    pyl.show()
    pyl.close()
    pyl.clf()

    return end, 

def test_CNN(X_train, X_test, y_test):
    ### get the model output classifications for the train and test sets
    preds_train = cn_model.predict(X_train, verbose=1)
    X_test = np.asarray(X_test).astype('float32')
    preds_test = cn_model.predict(X_test, verbose=1)

    # normalize too
    train_good_p = []
    test_good_p = []
    for p in preds_train:
        train_good_p.append(p[1])
    for p in preds_test:
        test_good_p.append(p[1])


    bins = np.linspace(0, 1, 100)
    pyl.hist(train_good_p, label = 'training set confidence', bins=bins, alpha=0.5, density=True) # normalize
    pyl.hist(test_good_p, label = 'test set confidence', bins=bins, alpha=0.5, density=True) # add transparency 
    pyl.xlabel('Good Star Confidence')
    pyl.ylabel('Count (normalized for each dataset)')
    pyl.legend(loc='best')
    pyl.show()
    pyl.close()
    pyl.clf()

    y_test_binary = keras.utils.np_utils.to_categorical(y_test, 2) # two diff y test binary
    results = cn_model.evaluate(X_test, y_test_binary, batch_size=batch_size)
    print("test loss, test acc:", results)

    zscale = ZScaleInterval()

    X_test = np.squeeze(X_test, axis=3)

    # plot confusion matrix
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
    fig2.savefig(file_dir+'/NN_plots/'+'NN_confusionMatrix' + str(end) + '.png')
    pyl.clf()
       
def main():

    balanced_data_method, data_load, size_of_data, \
    num_epochs, cutout_size, model_dir_name = get_user_input()

    if data_load == 'scratch':
        save_scratch_data(size_of_data, cutout_size, model_dir_name)

    cutouts, labels = load_data(cutout_size, model_dir_name) # assign fwhm and all too

    train_CNN(cutouts, labels)

    test_CNN()
    
if __name__ == '__main__':
    main()