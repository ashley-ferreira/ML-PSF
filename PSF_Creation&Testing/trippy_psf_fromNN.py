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

file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03071/HSC-R2/corr'
model_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr'

zscale = ZScaleInterval()



selected_file = sys.argv[1]
confidence_cutoff = 0.85 #conf = sys.argv[2]
stars_for_psf = int(sys.argv[2]) # stars_for_psf = 10 
indx = 1 # sys.argv[3] # 1 to look at good stars
inputFile = 'CORR-' + str(selected_file) + '.fits'
#inputFile = 'CORR-0218194-003.fits'
#outFile = file_dir+'/'+inputFile.replace('.fits', '_NN_savedFits.pickle')

# get img data 
with fits.open(file_dir+'/'+inputFile) as han:
    img_data = han[1].data
    header = han[0].header


outFile = file_dir+'/'+inputFile.replace('.fits', '_cutouts_savedFits.pickle')
with open(outFile, 'rb') as han:
    [stds, seconds, peaks, xs, ys, cutouts] = pickle.load(han)
                                        
model = keras.models.load_model(model_dir + '/Saved_Model/dec13_model')

xs_best = []
ys_best = []
cn_prob = []
cutouts_best = []

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
    if good_probability>float(confidence_cutoff):
        print(good_probability)
        xs_best.append(xs[i])
        ys_best.append(ys[i])
        cn_prob.append(good_probability)
        cutouts_best.append(cutouts[i])
        num_good_stars += 1

# order ones 

# [cutouts_best for _, cutouts_best in sorted(zip(good_probability, cutouts_best))] # does it leave probability sorted?
# final_NN_cutouts = cutouts_best[:stars_for_psf]
xs_final = []
ys_final = []

best_prob = sorted(cn_prob, reverse=True)[:stars_for_psf] # this doesnt sort acutal list right?

# need to sort x and y too
for i in range(len(cutouts_best)):
    print(cn_prob[i])
    if cn_prob[i] in best_prob: 
        # so will use more if probabilities tie
        xs_final.append(xs_best[i])
        ys_final.append(ys_best[i])
        (c1, c2) = zscale.get_limits(cutouts_best[i])
        normer3 = interval.ManualInterval(c1,c2)
        pyl.title('NN selected star, confidence=' + str(cn_prob[i]))
        pyl.imshow(normer3(cutouts_best[i]))
        pyl.show()
        pyl.close()



    
print('######################################################')
print('number of good stars selected for psf: ', num_good_stars)

xs_final = np.array(xs_final) 
ys_final = np.array(ys_final)
print(len(xs_final)) # how does this become 21?

starChooser=psfStarChooser.starChooser(img_data,
                                            xs_final,ys_final,
                                            xs_final*500,xs_final*1.0)

(goodFits, goodMeds, goodSTDs) = starChooser(30,200,noVisualSelection=True,autoTrim=False,
                                            bgRadius=15, quickFit = False,
                                            printStarInfo = True,
                                            repFact = 5, ftol=1.49012e-08)

goodPSF = psf.modelPSF(np.arange(61),np.arange(61), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
goodPSF.genLookupTable(img_data, goodFits[:,4], goodFits[:,5], verbose=False)

newPSFFile = file_dir+'/psfStars/'+inputFile.replace('.fits','.NN_PSF.fits')
print('Saving to', newPSFFile)
goodPSF.psfStore(newPSFFile, psfV2=True)


print('number of good stars selected for psf: ', num_good_stars)

fwhm = goodPSF.FWHM() ###this is the FWHM with lookuptable included
moffat_fwhm = goodPSF.FWHM(fromMoffatProfile=True)

# X, Y, flux, FWHM, moffat alpha, moffat beta, peak pixel value, CNN probability
with open('trippy_nn_psf_gen_troubleshooting', 'wb+') as han:
    #pickle.dump([cn_prob, xs_best, ys_best, fwhm, moffat_fwhm, goodFits[4]], goodFits[5], han)
    pickle.dump([cn_prob, goodFits], han)

''''
for i in range(num_good_stars):
    print(cn_prob[i], goodFits[i])
'''

(z1, z2) = zscale.get_limits(goodPSF.lookupTable)
normer = interval.ManualInterval(z1,z2)
pyl.imshow(normer(goodPSF.lookupTable))
title = 'ZScaled ' + inputFile.replace('.fits','.NN_PSF.fits')
pyl.title(title)

pyl.show()

# compare to origional, with and without zscale
comparePSF = file_dir+'/psfStars/'+inputFile.replace('.fits','.goodPSF.fits')
otherPSF = psf.modelPSF(restore=comparePSF)
(o1, o2) = zscale.get_limits(otherPSF.lookupTable)
normer2 = interval.ManualInterval(o1,o2)
pyl.imshow(normer2(otherPSF.lookupTable))
title = 'ZScaled ' + inputFile.replace('.fits','.goodPSF.fits')
pyl.title(title)
pyl.show()