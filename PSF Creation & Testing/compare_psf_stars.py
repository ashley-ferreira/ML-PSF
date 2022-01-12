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

import glob

from keras.utils import np_utils

file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr'
model_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr'

zscale = ZScaleInterval()

selected_file = sys.argv[1]
conf = sys.argv[2]
indx = sys.argv[3]
inputFile = 'CORR-' + str(selected_file) + '.fits'
#inputFile = 'CORR-0218194-003.fits'


outFile = file_dir+'/'+inputFile.replace('.fits', '_NN_savedFits.pickle')

# get img data 
with fits.open(file_dir+'/'+inputFile) as han:
    img_data = han[1].data.astype('float64')
    header = han[0].header
    
print(header)

# run star chooser

outFile = file_dir+'/'+inputFile.replace('.fits', '_cutouts_savedFits.pickle')
with open(outFile, 'rb') as han:
    [stds, seconds, peaks, xs, ys, cutouts] = pickle.load(han)
                                        
# run though my network and get - this model was good results!
model = keras.models.load_model(model_dir + '/Saved_Model/dec13_model_80epochs')

xs_best = []
ys_best = []
cn_prob = []

# load old std and mean and apply
with open(model_dir + '/regularization_data.pickle', 'rb') as han:
    [std, mean] = pickle.load(han)

cutouts -= mean
cutouts /= std
w_bad = np.where(np.isnan(cutouts))
cutouts[w_bad] = 0.0

output = model.predict(cutouts)
num_good_stars = 0 

for i in range(len(cutouts)): 
    # test on model 
    # output = model.predict(cutouts[i])
    good_probability = output[i][int(indx)]
    if good_probability > 0.5:
        print(good_probability)
    if good_probability>float(conf):
        pyl.title('NN selected star, confidence=' + str(good_probability))
        xs_best.append(xs[i])
        ys_best.append(ys[i])
        cn_prob.append(good_probability)
        num_good_stars += 1
        (c1, c2) = zscale.get_limits(cutouts[i])
        normer3 = interval.ManualInterval(c1,c2)
        #pyl.title('regularized + zscale')
        pyl.imshow(normer3(cutouts[i]))

        pyl.show()
        pyl.close()
        '''
        unreg = cutouts[i] + mean # change much?
        unreg = unreg * std
        
        (c1, c2) = zscale.get_limits(unreg)
        normer3 = interval.ManualInterval(c1,c2)
        pyl.imshow(normer3(unreg))
        pyl.show()
        pyl.close()
        '''
# compare to origional, with and without zscale


comparePSF = file_dir+'/psfStars/'+inputFile.replace('.fits','.goodPSF.fits')
with fits.open(comparePSF) as han:
    goodpsf_img_data = han[1].data
    header = han[0].header



cutoutWidth = 30
goodpsf_x = []
goodpsf_y = []
count = 0
print(header, type(header))
for e in header: # not both eh?
    if e[:5] == 'XSTAR':
        #print(e, type(header[count]))
        goodpsf_x.append(header[count])
    elif e[:5] =='YSTAR':
        #print(e, type(e))
        goodpsf_y.append(header[count])
    count += 1

print(goodpsf_x, goodpsf_y)
for i in range(len(goodpsf_x)):
    print('TOP 25 STAR')
    y_int = int(goodpsf_y[i])
    #print(y_int)
    x_int = int(goodpsf_x[i])
    #$print(x_int)
    cutout_goodpsf = img_data[y_int-cutoutWidth:y_int+cutoutWidth+1, x_int-cutoutWidth:x_int+cutoutWidth+1]
    #print(cutout_goodpsf) #this is empty
    #print(img_data)
    

    (c1, c2) = zscale.get_limits(cutout_goodpsf)
    normer4 = interval.ManualInterval(c1,c2)
    pyl.title('top 25 star')
    pyl.imshow(normer4(cutout_goodpsf))
    pyl.show()
    pyl.close()

    '''
    reg = cutout_goodpsf - mean
    reg = reg / std
    w_bad = np.where(np.isnan(reg))
    reg[w_bad] = 0.0
    (c1, c2) = zscale.get_limits(reg)
    normer5 = interval.ManualInterval(c1,c2)
    pyl.title('regularized + zscale')
    pyl.imshow(normer5(reg))
    pyl.show()
    pyl.close() 
    '''

xs_best = np.array(xs_best)
ys_best = np.array(ys_best)

starChooser=psfStarChooser.starChooser(img_data,
                                            xs_best,ys_best,
                                            xs_best*500,xs_best*1.0)

(goodFits, goodMeds, goodSTDs) = starChooser(30,200,noVisualSelection=True,autoTrim=False,
                                            bgRadius=15, quickFit = False,
                                            printStarInfo = True,
                                            repFact = 5, ftol=1.49012e-08)

goodPSF = psf.modelPSF(np.arange(61),np.arange(61), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
goodPSF.genLookupTable(img_data, goodFits[:,4], goodFits[:,5], verbose=False)


(z1, z2) = zscale.get_limits(goodPSF.lookupTable)
normer = interval.ManualInterval(z1,z2)
pyl.imshow(normer(goodPSF.lookupTable))
title = 'ZScaled ' + inputFile.replace('.fits','.NN_PSF.fits')
pyl.title(title)
pyl.show()

otherPSF = psf.modelPSF(restore=comparePSF)
(o1, o2) = zscale.get_limits(otherPSF.lookupTable)
normer2 = interval.ManualInterval(o1,o2)
pyl.imshow(normer2(otherPSF.lookupTable))
title = 'ZScaled ' + inputFile.replace('.fits','.goodPSF.fits')
pyl.title(title)
pyl.show()
# LAST THING IS TO ACUTLLY SHOW ZSCALED PSFS, this is great


# loop through and display
    #cutout_goodpsf = img_data[y_int-cutoutWidth:y_int+cutoutWidth+1, x_int-cutoutWidth:x_int+cutoutWidth+1]

'''
print(header)
otherPSF = psf.modelPSF(restore=comparePSF)
(o1, o2) = zscale.get_limits(otherPSF.lookupTable)
normer2 = interval.ManualInterval(o1,o2)
pyl.imshow(normer2(otherPSF.lookupTable))
title = 'ZScaled ' + inputFile.replace('.fits','.goodPSF.fits')
pyl.title(title)

pyl.show()
'''