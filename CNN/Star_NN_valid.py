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

# earlier, we set to train on selection from (219590,219620)
# here we validate on 219580 (and can also look at up to 88)

# load specific image (all CCDs) cutouts


# load model


# show stats analsys


# compare psfs