#### these are the major imports that you'll probably need
import os
import time
import sys
import heapq

import random
"""Import the basics: numpy, pandas, matplotlib, etc."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyl
import matplotlib.gridspec as gridspec
import pickle
"""Import keras and other ML tools"""
import tensorflow as tf
import keras

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv3D, Conv2D, MaxPool3D, MaxPool2D
from keras.layers.core import Dropout, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
"""Import scikit learn tools"""
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score # plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import class_weight
"""Import astropy libraries"""
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import interval
from astropy.visualization import interval, ZScaleInterval


from trippy import tzscale
from trippy.trippy_utils import expand2d, downSample2d

import glob

from keras.utils import np_utils

import numpy as np, astropy.io.fits as pyf,pylab as pyl
from trippy import psf, pill, psfStarChooser
from trippy import scamp,MCMCfit
import scipy as sci
from os import path
import sys
from astropy.visualization import interval, ZScaleInterval

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import interval

import numpy as np, astropy.io.fits as pyf,pylab as pyl
from trippy import psf, pill, psfStarChooser
from trippy import scamp,MCMCfit
import scipy as sci
from os import path
import sys
from astropy.visualization import interval, ZScaleInterval


import random
"""Import the basics: numpy, pandas, matplotlib, etc."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyl
import matplotlib.gridspec as gridspec
import pickle
"""Import keras and other ML tools"""
import tensorflow as tf
import keras

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv3D, Conv2D, MaxPool3D, MaxPool2D
from keras.layers.core import Dropout, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
#from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
"""Import scikit learn tools"""
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score # plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import class_weight
"""Import astropy libraries"""
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import interval

from trippy import tzscale
from trippy.trippy_utils import expand2d, downSample2d

#import seaborn as sns


import matplotlib as mpl

# GIT TEST

balanced_data_method = str(sys.argv[1]) # even or weight
data_load = str(sys.argv[2]) # can ask for specific presaved filename later
num_epochs = int(sys.argv[3])

# sset hyperparameters
batch_size = 16 # try diff batch size?
dropout_rate = 0.2
test_fraction = 0.05 # from 0.05
#num_epochs = 10
max_size = 111 # odd good

## initializing random seeds for reproducability
# tf.random.set_seed(1234)
# keras.utils.set_random_seed(1234)
np.random.seed(432)

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

#withheld_img = [219580, 219582, 219584, 219586, 219588]

zscale = ZScaleInterval()

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr'

if data_load == 'presaved':
    with open(file_dir + '/TRAIN_jan26_111_metadata_defaultLen.pickle', 'rb') as han:
        [cutouts, labels, xs, ys, fwhms, files] = pickle.load(han) # need count too?

    cutouts = np.asarray(cutouts).astype('float32')
    std = np.nanstd(cutouts)
    mean = np.nanmean(cutouts)
    cutouts -= mean
    cutouts /= std
    # And just to be sure you aren’t picking up any bad values, after regularization:
    w_bad = np.where(np.isnan(cutouts))
    cutouts[w_bad] = 0.0

    with open(file_dir + '/regularization_data.pickle', 'wb+') as han:
        pickle.dump([std, mean], han)

elif data_load == 'scratch':
    
    size_of_data = int(sys.argv[4])//2
    #cutout_len = int(sys.argv[5])

    class BreakException(Exception):
        pass  

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
                ''' 
                (c1, c2) = zscale.get_limits(cutout)
                normer4 = interval.ManualInterval(c1,c2)
                pyl.imshow(normer4(cutout))
                pyl.show()
                pyl.close()
                '''
                #print(inputFile)
                imgFile = int(inputFile[6:12]) #'CORR-0' + str(k) + '-' + num_str + '.fits'
                print(imgFile)
                # TEMPORARY
                if len(cutout) > 0:# and not(imgFile in withheld_img):
                    #l1 = len(cutout[0])
                    #l2 = len(cutout[:][0])
                    #print(l1,l2)
                    #cutout = np.array(cutout)
                    #if l1 == 111 and l2 == 111:
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
                        else:
                            print('ERROR: label is not 1 or 0')
                        #print(cutout.shape)
                    else:
                        continue

                else:
                    print('Saving cutouts from ',imgFile,' for validation')
    except BreakException:
        pass

    # make sure there are more good stars then bad ones?
    if len(good_cutouts)>len(bad_cutouts):
        print('############  MORE BAD STARS THAN GOOD STARS #############')

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

    good_cutouts = np.array(good_cutouts)#, dtype=object)
    print(good_cutouts.shape)
    good_cutouts = np.expand_dims(good_cutouts, axis=3)
    print(good_cutouts.shape)

    # add label 1
    label_good = np.ones(num_good_cutouts)

    bad_cutouts = np.array(bad_cutouts, dtype=object) #new addition, unsure

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

    elif balanced_data_method == 'weight': # NEED TO ADD METADATA
        label_bad = np.zeros(len(bad_cutouts))
        bad_cutouts = np.expand_dims(bad_cutouts, axis=3)
        #class_weights = {0: , 1: }

    else:
        print('invalidid argv[1] must be: "even, "weights"')


    # combine arrays 
    cutouts = np.concatenate((good_cutouts, bad_cutouts))
    fwhms = np.concatenate((good_fwhm_arr, random_bad_fwhm_arr))
    files = np.concatenate((good_inputFile_arr, random_bad_fwhm_arr))
    xs = np.concatenate((good_x_arr, random_bad_x_arr))
    ys = np.concatenate((good_y_arr, random_bad_y_arr))

    # make label array for all
    labels = np.concatenate((label_good, label_bad))

    # mix these arrays (needed?)
                
    print(str(files_counted) + ' processed so far')
    print(str(len(cutouts)) + ' files used')

    valid_fraction = 0.1
    skf = StratifiedShuffleSplit(n_splits=1, test_size=valid_fraction)#, random_state=41)
    skf.split(cutouts, labels)

    for used_index, withheld_index in skf.split(cutouts, labels): # an issue with split here
        used_cutouts, withheld_cutouts = cutouts[used_index], cutouts[withheld_index]
        used_labels, withheld_labels = labels[used_index], labels[withheld_index]
        used_xs, withheld_xs = xs[used_index], xs[withheld_index]
        used_ys, withheld_ys = ys[used_index], ys[withheld_index]
        used_files, withheld_files = files[used_index], files[withheld_index]
        used_fwhms, withheld_fwhms = fwhms[used_index], fwhms[withheld_index]

    with open(file_dir + '/USED_jan27_50k_' + str(max_size) + '_metadata_defaultLen.pickle', 'wb+') as han:
        pickle.dump([used_cutouts, used_labels, used_xs, used_ys, used_fwhms, used_files], han)

    with open(file_dir + '/WITHHELD_jan27_50k_' + str(max_size) + '_metadata_defaultLen.pickle', 'wb+') as han:
        pickle.dump([withheld_cutouts, withheld_labels, withheld_xs, withheld_ys, withheld_fwhms, withheld_files], han)

    used_cutouts = np.asarray(used_cutouts).astype('float32')
    std = np.nanstd(used_cutouts)
    mean = np.nanmean(used_cutouts)
    used_cutouts -= mean
    used_cutouts /= std
    # And just to be sure you aren’t picking up any bad values, after regularization:
    w_bad = np.where(np.isnan(used_cutouts)) # how is this  a float?
    used_cutouts[w_bad] = 0.0

    with open(file_dir + '/regularization_data.pickle', 'wb+') as han:
        pickle.dump([std, mean], han)

else: 
    print('invalid data load method, must be "scratch" or "presaved"')

### now divide the cutouts array into training and testing datasets.
skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction)#, random_state=41)
print(skf)
skf.split(used_cutouts, labels)

#print(cutouts.shape) # why does it need both?
for train_index, test_index in skf.split(used_cutouts, used_labels):
    X_train, X_test = used_cutouts[train_index], used_cutouts[test_index]
    y_train, y_test = used_labels[train_index], used_labels[test_index]
    xs_train, xs_test = used_xs[train_index], used_xs[test_index]
    files_train, files_test = used_files[train_index], used_files[test_index]
    fwhms_train, fwhms_test = used_fwhms[train_index], used_fwhms[test_index]

### define the CNN
# below is a network I used for KBO classification from image data.
# you'll need to modify this to use 2D convolutions, rather than 3D.
# the Maxpool lines will also need to use axa kernels rather than axaxa
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
    model.add(Dense(2, activation='softmax')) # 2 instead of unique labs - temp change
    #model.add(Activation("softmax"))

    return model

training_labels = y_train
unique_labs = len(np.unique(training_labels))
print(unique_labels, y_test)
y_train_binary = keras.utils.np_utils.to_categorical(y_train, 2) #unique_labels)
y_test_binary = keras.utils.np_utils.to_categorical(y_test, 2) #unique_labels)

### train the model!
cn_model = convnet_model(X_train.shape[1:], training_labels=training_labels, unique_labs=2)
#cn_model = convnet_model(X_train.shape[1:], y_train)
cn_model.summary()

opt = Adam(learning_rate=0.001)
cn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])#, learning_rate=0.1)


checkpointer = ModelCheckpoint('keras_convnet_model_test.h5', verbose=1)
#early_stopper = EarlyStopping(monitor='loss', patience=2, verbose=1)
early_stopper = EarlyStopping(monitor='categorical_accuracy', patience=10, verbose=1)

start = time.time()
X_train = np.asarray(X_train).astype('float32')
y_train_binary = np.asarray(y_train_binary).astype('float32')

if balanced_data_method == 'even':
    classifier = cn_model.fit(X_train, y_train_binary, epochs=num_epochs, batch_size=batch_size)#, callbacks=[checkpointer]) # validation_split = 

elif balanced_data_method == 'weight':
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    #class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_binary), y_train_binary)
    #class_weights = {1: len(bad_cutouts)/len(good_cutouts), 0: 1.} # there are more complex ways of doing this
    neg = len(bad_cutouts)
    pos = len(good_cutouts)
    total = pos + neg
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    classifier = cn_model.fit(X_train, y_train_binary, epochs=num_epochs, batch_size=batch_size, class_weight=class_weight) #callbacks=[checkpointer]


end = time.time()
print('Process completed in', round(end-start, 2), ' seconds')
# save details of model and regulatization data in here too
cn_model.save(file_dir + '/Saved_Model/model_jan27_50k') # + str(end))


"""
Plot accuracy/loss versus epoch
"""

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

results = cn_model.evaluate(X_test, y_test_binary, batch_size=batch_size)
print("test loss, test acc:", results)

zscale = ZScaleInterval()

X_test = np.squeeze(X_test, axis=3)
print(X_test.shape) # doesnt look squeezed?


# plot confusion matrix
fig2 = pyl.figure()

y_test_binary = np.argmax(y_test_binary, axis = 1)
preds_test_binary = np.argmax(preds_test, axis = 1)

cm = confusion_matrix(y_test_binary, preds_test_binary)
pyl.matshow(cm)

for (i, j), z in np.ndenumerate(cm):
    pyl.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

pyl.title('Confusion matrix')
pyl.colorbar(cmap=mpl.cm.cool)
pyl.xlabel('Predicted labels')
pyl.ylabel('True labels')
pyl.show()
fig2.savefig(file_dir+'/NN_plots/'+'NN_confusionMatrix' + str(end) + '.png')
pyl.clf()

fwhms_test_misclass = []
for i in range(len(preds_test)):
    #print(y_test[i])
    #print(preds_test[i])
    #pyl.imshow(X_test[i])
    #pyl.show()
    #pyl.close()
    
    if y_test[i] == 1 and preds_test[i][0] > 0.5:
        fwhms_test_misclass.append(fwhms_test[i])
        #print(fwhms_test[i])
        #    print('GOOD STAR LABEL')
        #    print(preds_test[i])
        # been regularized so diff?
        #(c1, c2) = zscale.get_limits(y_test[i])
        #normer3 = interval.ManualInterval(c1,c2)
        #pyl.title('labeled good star, predicted bad star at conf=' + str(preds_test[i][1]))
        #pyl.imshow(X_test[i])
        #pyl.show()
        #pyl.close()

    elif y_test[i] == 0 and preds_test[i][1] > 0.5:
        fwhms_test_misclass.append(fwhms_test[i])
        #print(fwhms_test[i]) # noen in this class?
        #(c1, c2) = zscale.get_limits(X_test[i])
        #normer5 = interval.ManualInterval(c1,c2)
        #pyl.title('labeled bad star, predicted good star at conf=' + str(preds_test[i][1])) # so great you already have this
        #pyl.imshow(normer5(X_test[i]))
        #pyl.show()
        #pyl.close()
    
print(fwhms_test)    
pyl.hist(fwhms_test, label = 'FWHM of full test set', bins='auto', alpha=0.5) 
pyl.hist(fwhms_test_misclass, label = 'FWHM of misclassed test set', bins='auto', alpha=0.5) 
pyl.xlabel('FWHM')
pyl.ylabel('Count')
pyl.legend(loc='best')
pyl.show()
pyl.close()
pyl.clf()

misclass_80p = 0
good_class_80p = 0
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
    elif y_test[i] == 1 and preds_test[i][1] > 0.8:
        good_class_80p += 1

print('number of misclassed good stars above 80 percent confidence:', misclass_80p)
print('number of correctly classified good stars above 80 percent confidence', good_class_80p)
print('out of total test set size (50/50 split):', len(preds_test))