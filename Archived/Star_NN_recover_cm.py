import os
import time
import sys

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

batch_size = 16 
dropout_rate = 0.2
test_fraction = 0.05 
np.random.seed(432)

good_cutouts = [] # label 1
bad_cutouts = [] # label 0

zscale = ZScaleInterval()

file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr'

if True:
    with open(file_dir + '/all_61_noreg_data.pickle', 'rb') as han:
        [cutouts, labels] = pickle.load(han)

    std = np.nanstd(cutouts)
    mean = np.nanmean(cutouts)
    cutouts -= mean
    cutouts /= std
    w_bad = np.where(np.isnan(cutouts))
    cutouts[w_bad] = 0.0

    with open(file_dir + '/regularization_data.pickle', 'wb+') as han:
        pickle.dump([std, mean], han)

skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction)
print(skf)
skf.split(cutouts, labels)

print(cutouts.shape) 
for train_index, test_index in skf.split(cutouts, labels):
    X_train, X_test = cutouts[train_index], cutouts[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

cn_model = keras.models.load_model(file_dir + '/Saved_Model/model1639629151.572077')

preds_test = cn_model.predict(X_test, verbose=1)

y_train_binary = keras.utils.np_utils.to_categorical(y_train, 2) 
y_test_binary = keras.utils.np_utils.to_categorical(y_test, 2) 

results = cn_model.evaluate(X_test, y_test_binary, batch_size=batch_size)
print("test loss, test acc:", results)

zscale = ZScaleInterval()

X_test = np.squeeze(X_test, axis=3)
print(X_test.shape)

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
pyl.clf()
