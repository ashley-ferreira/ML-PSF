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

def get_user_input():
    val = input("Change default values (Y/N): ")
balanced_data_method = str(sys.argv[1]) # even or weight
data_load = str(sys.argv[2]) # can ask for specific presaved filename later
num_epochs = int(sys.argv[3])

####section for setting up some flags and hyperparameters
batch_size = 16 # try diff batch size?
dropout_rate = 0.2
test_fraction = 0.05 # from 0.05
#num_epochs = 10
max_size = 111 # odd good

size_of_data = int(sys.argv[4])//2
#cutout_len = int(sys.argv[5])

## initializing random seeds for reproducability
# tf.random.set_seed(1234)
# keras.utils.set_random_seed(1234)
np.random.seed(432)

zscale = ZScaleInterval()


file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr'

def load_presaved_data(data_file):
    with open(file_dir + '/jan18_111_metadata_defaultLen.pickle', 'rb') as han:
        [cutouts, labels, xs, ys, fwhm, files] = pickle.load(han) 

    cutouts = np.asarray(cutouts).astype('float32')
    std = np.nanstd(cutouts)
    mean = np.nanmean(cutouts)
    cutouts -= mean
    cutouts /= std
    w_bad = np.where(np.isnan(cutouts))
    cutouts[w_bad] = 0.0

    #with open(file_dir + '/regularization_data.pickle', 'wb+') as han:
    #    pickle.dump([std, mean], han)

def save_scratch_data(size_of_data, data_file):

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

    files_counted = 0
    try:
        for filename in os.listdir(file_dir+ '/NN_data_metadata_111'):
            if filename.endswith("metadata_cutoutData.pickle"):
                #print(files_counted, size_of_data)
                if files_counted >= size_of_data:
                    break
                print(files_counted, size_of_data)
                #print('file being processed: ', filename)

                with open(file_dir + '/NN_data_metadata_111/' + filename, 'rb') as f:
                    [n, cutout, label, y, x, fwhm, inputFile] = pickle.load(f)

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
                        else:
                            print('ERROR: label is not 1 or 0, excluding cutout')
                    else:
                        continue
    except Exception as e: 
        print('FAILURE')
        print(e) 
        pass   

    # make sure there are more good stars then bad ones?
    if len(good_cutouts)>len(bad_cutouts):
        print('ERROR: MORE BAD STARS THAN GOOD STARS')
        return 0

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

    with open(file_dir + '/jan19_' + str(max_size) + '_metadata_defaultLen.pickle', 'wb+') as han:
        pickle.dump([cutouts, labels, xs, ys, fwhm, files], han)

    return 1

def load_data(load_method, datset_size, data_file, test_fraction): # use global variables?
    if load_method == 'scratch':
        save_scratch_data(dataset_size, data_file)
    load_presaved_data()
    #elif load_method == 'presaved':
    #    load_presaved_data()
    #else: 
    #    print('invalid data load method, must be "scratch" or "presaved"')
    #    return 0

    return cutouts, labels, xs, ys, # load from file?

    ### now divide the cutouts array into training and testing datasets.
    skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction)#, random_state=41)
    print(skf)
    skf.split(cutouts, labels)

    print(cutouts.shape) # why does it need both?
    for train_index, test_index in skf.split(cutouts, labels):
        X_train, X_test = cutouts[train_index], cutouts[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        xs_train, xs_test = xs[train_index], xs[test_index]
        files_train, files_test = files[train_index], files[test_index]
        fwhms_train, fwhms_test = fwhms[train_index], fwhms[test_index]

    return  X_train, X_test, y_train, y_test, files_train, files_test, files_train, files_test, fwhms_train, fwhms_test

### define the CNN
def convnet_model(input_shape, training_labels, unique_labs, dropout_rate=dropout_rate):

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
    
    #unique_labs = len(np.unique(y_train))
    #print(unique_labels)
    y_train_binary = keras.utils.np_utils.to_categorical(y_train, 2) #unique_labels)

    ### train the model!
    cn_model = convnet_model(X_train.shape[1:], training_labels=y_train, unique_labs=2)
    #cn_model = convnet_model(X_train.shape[1:], y_train)
    cn_model.summary()

    opt = Adam(learning_rate=0.001)
    cn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])#, learning_rate=0.1)

    start = time.time()
    X_train = np.asarray(X_train).astype('float32')
    y_train_binary = np.asarray(y_train_binary).astype('float32')

    classifier = cn_model.fit(X_train, y_train_binary, epochs=num_epochs, batch_size=batch_size)

    end = time.time()
    print('Process completed in', round(end-start, 2), ' seconds')
    # save details of model and regulatization data in here too
    today = date.today()
    date_trained = today.strftime("%b-%d-%Y")
    cn_model.save(file_dir + '/Saved_Model_/model_' + str(end))
    with open(file_dir + '/Saved_Model/jan19_' + str(max_size) + '_metadata_defaultLen.pickle', 'wb+') as han:
            pickle.dump([cutouts, labels, xs, ys, fwhm, files], han)


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

    #print('Top 25 (or tied within) predicted good stars that are labelled wrong:') # diff images so hard, seperate this
    #z = heapq.nlargest(25, preds_test[i][1])
    #print(z)
    #misclass_25 = 0
    #for i in range(len(preds_test)):
    #    # need top 25 confidence
    #    if y_test[i] == 0 and preds_test[i][1] in z:
    #        (c1, c2) = zscale.get_limits(X_test[i])
    #        normer5 = interval.ManualInterval(c1,c2)
    #        pyl.title('labeled bad star, predicted good star at conf=' + str(preds_test[i][1])) # so great you already have this
    #        pyl.imshow(normer5(X_test[i]))
    #        pyl.show()
    #        pyl.close()
    #        misclass_25 += 1

    #print('Misclassed stars in top 25', misclass_25)
    misclass_80p = 0
    for i in range(len(preds_test)):
        # need top 25 confidence
        if y_test[i] == 0 and preds_test[i][1] > 0.8:
            (c1, c2) = zscale.get_limits(X_test[i])
            normer5 = interval.ManualInterval(c1,c2)
            pyl.title('labeled bad star, predicted good star at conf=' + str(preds_test[i][1])) # so great you already have this
            pyl.imshow(normer5(X_test[i]))
            pyl.show()
            pyl.close()
            misclass_80p += 1

    print('Misclassed good stars above 80 percent confidence', misclass_80p)
       
def main():
    get_user_input()
    load_data()
    train_CNN()
    test_CNN()
    
if __name__ == '__main__':
    main()