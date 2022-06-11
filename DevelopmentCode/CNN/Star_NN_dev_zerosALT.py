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

from resnet_model_revised import convnet_model_resnet

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
    if not(os.path.exists(model_dir_name)):
        os.mkdir(model_dir_name)
    plots_dir = model_dir_name + 'plots/'
    if not(os.path.exists(plots_dir)):
        os.mkdir(plots_dir)
    submodels_dir = model_dir_name + 'models_each_10epochs_RESNET64/'
    if not(os.path.exists(submodels_dir)):
        os.mkdir(submodels_dir)
    
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

    N_bad = 967002 #970000 # then delete extra zero ones, can make this number proportional to good star number
    bad_arr = np.zeros((N_bad, 111, 111), dtype='float') # correct shape? dont expand dims later
    print(bad_arr.shape)
    # specific float kind?
    good_counted = 0
    bad_counted = 0
    max_num_good = size_of_data//2
    i=0
    try:
        for filename in os.listdir(data_dir):
            if good_counted < max_num_good:
                if filename.endswith('_cutoutData.pickle') and os.path.getsize(data_dir + filename) > 0:
                    print(good_counted, 'good stars out of max number needed', max_num_good, 'processed')
                    print(bad_counted, 'bad stars out of max number number needed', max_num_good, 'processed')
                    print('file being processed: ', filename)

                    with open(data_dir + filename, 'rb') as f:
                        [n, cutout, label, y, x, fwhm, inputFile] = pickle.load(f)
                        inf_or_nan = np.isfinite(cutout)
                        if False in inf_or_nan:
                            pass
                        
                        elif cutout.shape == (cutout_size, cutout_size):# and np.isfinite(cutout):
                            if cutout.min() < -2000 or cutout.max() > 130000:
                                pass
                            else:
                                if cutout.min() < -200 or cutout.max() > 65536:
                                    label = 0
                                if label == 1:
                                    good_x_lst.append(x)
                                    good_y_lst.append(y)
                                    good_fwhm_lst.append(fwhm)
                                    good_inputFile_lst.append(inputFile)
                                    good_cutouts.append(cutout)
                                    good_counted += 1
                                # short term sol, long term sol is already decide
                                # random incidies
                                elif label == 0:
                                    bad_x_lst.append(x)
                                    bad_y_lst.append(y)
                                    bad_fwhm_lst.append(fwhm)
                                    bad_inputFile_lst.append(inputFile)
                                    bad_arr[bad_counted,:,:] = np.copy(cutout)#expand dims later?
                                    bad_counted += 1
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

    
    print('all cutouts loaded')

    # make sure there are more good stars then bad ones
    num_good_cutouts = len(good_cutouts)
    num_bad_cutouts = len(bad_arr)
    if num_good_cutouts > num_bad_cutouts:
        print('ERROR: MORE GOOD STARS THAN BAD STARS')

    print('converting data to arrays...')
    # mem save ideas (use asrray instead of array? no copy)
    # should you call it the same thing not list and arr?
    # split up this conversion to after saving
    # print to see which line has real issue
    # append values directly to an array?

    # convert lists to arrays and delete lists
    good_x_arr = np.asarray(good_x_lst)
    del good_x_lst
    good_y_arr = np.asarray(good_y_lst)
    del good_y_lst
    bad_x_arr = np.asarray(bad_x_lst)
    del bad_x_lst
    bad_y_arr = np.asarray(bad_y_lst)
    del bad_y_lst
    print('x and y values converted')

    good_fwhm_arr = np.asarray(good_fwhm_lst)
    del good_fwhm_lst
    bad_fwhm_arr = np.asarray(bad_fwhm_lst)
    del bad_fwhm_lst
    print('FWHMs converted')

    good_inputFile_arr = np.asarray(good_inputFile_lst)
    del good_inputFile_lst
    bad_inputFile_arr = np.asarray(bad_inputFile_lst)
    del bad_inputFile_lst
    print('filenames converted')   

    # convert cutout lists to arrays
    good_cutouts = np.asarray(good_cutouts) 
    print('good cutouts converted')

    #np.save(outfile, bad_cutouts)
    #del bad_cutouts
    #outfile.seek(0)
    #bad_cutouts = np.load(outfile)

    # convert each part of list to array seperately and del as you go?
    # indicies is good but not for nans but you can do that 
    # later on once its more balanced
    # you can take out zeros of array in future?

    #bad_cutouts = np.asarray(bad_cutouts)#, dtype=np.float16)#, dtype=object) .astype(np.float16)
    # make into arrays directly earlier and combine here? save good cutouts
    # can also initialize zeros and fill in

    # REMOVE ZEROS HERE (THIS CREATES COPY)
    #np.delete(bad_arr, range(bad_counted,N_bad))
    print('bad cutouts converted') 

    print('all data successfully converted to arrays')

    # expand dims of good cutouts (done later for bad cutouts)
    good_cutouts = np.expand_dims(good_cutouts, axis=3) #worked before, add good cutouts same as bad
    print('good cutouts dimentions expanded')

    # add label 1 to good cutouts 
    label_good = np.ones(num_good_cutouts)
    print('good cutouts labels created')


    # more bad cutouts than good cutouts
    if balanced_data_method == 'even':
        # more mem efficient way of below?
        number_of_rows = bad_arr.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_good_cutouts, replace=False)
        random_bad_cutouts = bad_arr[random_indices, :]
        del bad_cutouts
        # can expand dim at read in too
        random_bad_cutouts = np.expand_dims(random_bad_cutouts, axis=3)
        
        # better way to have foresign on that number?
        random_bad_x_arr = bad_x_arr[random_indices]
        random_bad_y_arr = bad_y_arr[random_indices]
        random_bad_fwhm_arr = bad_fwhm_arr[random_indices]
        random_bad_inputFile_arr = bad_inputFile_arr[random_indices]

        # add label 0
        label_bad = np.zeros(num_good_cutouts)

        print('# good stars and # bad stars are now balanced')

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


    # combine arrays and delete old bits
    cutouts = np.concatenate((good_cutouts, random_bad_cutouts))
    del good_cutouts
    del random_bad_cutouts

    fwhms = np.concatenate((good_fwhm_arr, random_bad_fwhm_arr))
    del good_fwhm_arr
    del random_bad_fwhm_arr

    files = np.concatenate((good_inputFile_arr, random_bad_inputFile_arr))
    del good_inputFile_arr
    del random_bad_inputFile_arr

    xs = np.concatenate((good_x_arr, random_bad_x_arr))
    del good_x_arr
    del random_bad_x_arr

    ys = np.concatenate((good_y_arr, random_bad_y_arr))
    del good_y_arr
    del random_bad_y_arr

    # make label array for all
    labels = np.concatenate((label_good, label_bad))
    del label_good
    del label_bad
    print(str(len(cutouts)) + ' files used')

    print('good and bad star arrays have been combined')

    skf_v = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction)
    skf_v.split(cutouts, labels)

    for used_index, withheld_index in skf_v.split(cutouts, labels): 
        used_cutouts, withheld_cutouts = cutouts[used_index], cutouts[withheld_index] 
        del cutouts 

        used_labels, withheld_labels = labels[used_index], labels[withheld_index]
        del labels

        used_xs, withheld_xs = xs[used_index], xs[withheld_index]
        del xs  

        used_ys, withheld_ys = ys[used_index], ys[withheld_index]
        del ys 

        used_files, withheld_files = files[used_index], files[withheld_index]
        del files 

        used_fwhms, withheld_fwhms = fwhms[used_index], fwhms[withheld_index]
        del fwhm

    print('good and bad stars have been randomly split into the train and validate sets')

    with open(model_dir_name + 'USED_' + str(cutout_size) + '_presaved_data.pickle', 'wb+') as han:
        pickle.dump([used_cutouts, used_labels, used_xs, used_ys, used_fwhms, used_files], han, protocol=4)

    with open(model_dir_name + 'WITHHELD_' + str(cutout_size) + '_presaved_data.pickle', 'wb+') as han:
        pickle.dump([withheld_cutouts, withheld_labels, withheld_xs, withheld_ys, withheld_fwhms, withheld_files], han, protocol=4)


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
    print('Begin data loading...')
    with open(model_dir_name + 'USED_' + str(cutout_size) + '_presaved_data.pickle', 'rb') as han:
        [cutouts, labels, xs, ys, fwhms, files] = pickle.load(han) 
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

    cutouts = np.asarray(cutouts).astype('float32')
    std = np.nanstd(cutouts)
    mean = np.nanmean(cutouts)
    cutouts = regularize(cutouts, mean, std)
    print('Data all regulatized')
    with open(model_dir_name + 'regularization_data.pickle', 'wb+') as han:
        pickle.dump([std, mean], han)

    return [cutouts, labels, xs, ys, fwhms, files]


def train_CNN(model_dir_name, num_epochs, data):
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
    sub_mod_dir = model_dir_name + 'models_each_10epochs_RESNET64/'#'models_lesslay16_256_lr=0.001_drop=0.2_split=0.2/'
    if not(os.path.exists(sub_mod_dir)):
        os.mkdir(sub_mod_dir)
    class CustomSaver(keras.callbacks.Callback):
        '''
        taken from:
        https://stackoverflow.com/questions/54323960/save-keras-model-at-specific-epochs
        '''
        def on_epoch_end(self, epoch, logs={}):
            if (epoch+1) % 10 == 0:
                self.model.save(sub_mod_dir + "model_{}".format(epoch+1))

    # unpack presaved data
    cutouts, labels, xs, ys, fwhms, files = data[0], data[1], data[2], data[3], data[4], data[5]

    # section for setting up some flags and hyperparameters
    batch_size = 256 # up from 16 --> 1024 --> 32 --> 256
    dropout_rate = 0.2
    test_fraction = 0.2 # from 0.05
    learning_rate = 0.001#01 # down from 0.001 --> 0.0005 --> 0.00001

    ### now divide the cutouts array into training and testing datasets.
    skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=0)
    print(skf)
    skf.split(cutouts, labels)

    for train_index, test_index in skf.split(cutouts, labels):
        X_train, X_test = cutouts[train_index], cutouts[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        xs_train, xs_test = xs[train_index], xs[test_index]
        ys_train, ys_test = xs[train_index], xs[test_index]
        files_train, files_test = files[train_index], files[test_index]
        fwhms_train, fwhms_test = fwhms[train_index], fwhms[test_index]
    print('Data split into training and testing')
    unique_labs = len(np.unique(y_train)) # should be 2
    y_train_binary = keras.utils.np_utils.to_categorical(y_train, unique_labs)

    # train the model
    cn_model = convnet_model_resnet(X_train.shape[1:], unique_labels=unique_labs, dropout_rate=dropout_rate)
    cn_model.summary()

    opt = Adam(learning_rate=learning_rate) 
    cn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

    start = time.time()
    X_train = np.asarray(X_train).astype('float32')
    y_train_binary = np.asarray(y_train_binary).astype('float32')

    # REDUNDANT
    y_test_binary = keras.utils.np_utils.to_categorical(y_test, unique_labs)
    y_test_binary = np.asarray(y_test_binary).astype('float32')

    print('Model initialized and prepped, begin training...')

    # add saver for every 10 epochs
    saver = CustomSaver()
    classifier = cn_model.fit(X_train, y_train_binary, epochs=num_epochs, batch_size=batch_size, callbacks=[saver], validation_data=(X_test, y_test_binary))

    end = time.time()
    print('Process completed in', round(end-start, 2), ' seconds')

    # save trained model 
    cn_model.save(model_dir_name + 'model_traintime=' + str(end))

    # plot accuracy/loss versus epoch
    fig1 = pyl.figure(figsize=(10,3))

    ax1 = pyl.subplot(121)
    ax1.plot(classifier.history['accuracy'], color='darkslategray', linewidth=2, label='training')
    ax1.plot(classifier.history['val_accuracy'], linewidth=2, label='testing') # azure linewidth=2, label='testing')
    ax1.legend()
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')

    ax2 = pyl.subplot(122)
    ax2.plot(classifier.history['loss'], color='crimson', linewidth=2, label='training')
    ax2.plot(classifier.history['val_loss'], linewidth=2, label='testing')
    ax2.legend()
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')

    fig1.savefig(model_dir_name +'/plots/'+'NN_training_history_model_traintime=' + str(end) + '.png')

    pyl.show()
    pyl.close()
    pyl.clf()

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
    fig2.savefig(model_dir_name +'plots/'+'NN_confusion_matrix.png') # add model secific time here
    pyl.close()
    pyl.clf()

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