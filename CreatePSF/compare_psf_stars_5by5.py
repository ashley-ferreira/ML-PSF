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

import matplotlib.pyplot as plt

file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03068/HSC-R2/corr'
model_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr'

zscale = ZScaleInterval()
fixed_cutout_len = 111

selected_file = sys.argv[1]
#conf = sys.argv[2]
#indx = sys.argv[3]
inputFile = 'CORR-' + str(selected_file) + '.fits'
#inputFile = 'CORR-0218194-003.fits'


outFile = file_dir+'/'+inputFile.replace('.fits', '_NN_savedFits.pickle')

# get img data 
with fits.open(file_dir+'/'+inputFile) as han:
    img_data = han[1].data.astype('float64')
    header = han[0].header
    
print(header)

# run star chooser

outFile = file_dir+'/'+inputFile.replace('.fits', str(fixed_cutout_len) + '_metadata_cutouts_savedFits.pickle')
with open(outFile, 'rb') as han:
    #[stds, seconds, peaks, xs, ys, cutouts] = pickle.load(han)
    [std, seconds, peaks, xs, ys, cutouts, fwhm, inputFile] = pickle.load(han)

print(cutouts.shape)
                                        
# run though my network and get - this model was good results!
model = keras.models.load_model(model_dir + '/Saved_Model/model_jan27_25k_250epochs')

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
indx = 1

for i in range(len(cutouts)):
    good_probability = output[i][int(indx)]
    cn_prob.append(good_probability)
    num_good_stars += 1 #NOT RIGHT


best_prob = sorted(cn_prob, reverse=True)[:25] # consider sorting cutouts by confidence
print('lowest confidence in top 25', best_prob[24])
fig, axs = plt.subplots(5,5,figsize=(5*5, 5*5))
axs = axs.ravel()
plt.title('NN selected top 25 stars:' + inputFile, x=-1.5, y=5) #hasnt changed location?
plotted_stars = 0
for i in range(len(cutouts)): # CURRUPTED FILE? yeah normal ones arent working?
    #pyl.imshow(cutouts[i])
    if plotted_stars < 25:
        good_probability = output[i][1]#int(indx)]
        #if cn_prob[i] in best_prob: 
        if good_probability in best_prob:       
            xs_best.append(xs[i])
            ys_best.append(ys[i])
            cn_prob.append(good_probability)
            
            (c1, c2) = zscale.get_limits(cutouts[i])
            normer3 = interval.ManualInterval(c1,c2)
            axs[plotted_stars].imshow(normer3(cutouts[i]))
            axs[plotted_stars].set_xticks([])
            axs[plotted_stars].set_yticks([])
            axs[plotted_stars].text(good_probability,0.1,0.1)
            plotted_stars += 1 
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
# dont show axis labels
# do show images? issue with non good psf ones
# title on top
# lowest confidence in
xs_threshold = []
ys_threshold = []
top15_prob = best_prob[10] # find lowerst part and all above that make it
print(top15_prob)
if top15_prob < 0.95:
    print('Neural Network not confident enough')
    sys.exit()
else:
    fig, axs = plt.subplots(5,5,figsize=(5*5, 5*5))
    axs = axs.ravel()
    plt.title('NN selected top ~10 stars above threshold:' + inputFile, x=-1.5, y=5) #hasnt changed location?
    plotted_stars = 0
    for i in range(len(cutouts)): # only 8?
        if plotted_stars < 15:
            good_probability = output[i][1]
            if good_probability > top15_prob:
                print(good_probability, plotted_stars)       
                xs_threshold.append(xs[i])
                ys_threshold.append(ys[i])
                cn_prob.append(good_probability)
                
                (c1, c2) = zscale.get_limits(cutouts[i])
                normer3 = interval.ManualInterval(c1,c2)
                axs[plotted_stars].imshow(normer3(cutouts[i]))
                axs[plotted_stars].set_xticks([])
                axs[plotted_stars].set_yticks([])
                plotted_stars += 1 
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


comparePSF = file_dir+'/psfStars/'+inputFile.replace('.fits','.metadata_goodPSF.fits')
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

fig, axs = plt.subplots(5,5,figsize=(5*5, 5*5))
axs = axs.ravel()
plt.title('goodPSF selected top 25 stars:' + inputFile, x=-1.5, y=5)
cutoutWidth = 55
for i in range(len(goodpsf_x)):
    y_int = int(goodpsf_y[i])
    #print(y_int)
    x_int = int(goodpsf_x[i])
    #$print(x_int)
    cutout_goodpsf = img_data[y_int-cutoutWidth:y_int+cutoutWidth+1, x_int-cutoutWidth:x_int+cutoutWidth+1]
    print(cutout_goodpsf.shape) # check
    cutout_goodpsf -= mean
    cutout_goodpsf /= std
    w_bad = np.where(np.isnan(cutout_goodpsf))
    cutout_goodpsf[w_bad] = 0.0
    (c1, c2) = zscale.get_limits(cutout_goodpsf)
    normer4 = interval.ManualInterval(c1,c2)
    axs[i].imshow(normer4(cutout_goodpsf))
    axs[i].set_xticks([])
    axs[i].set_yticks([])
plt.subplots_adjust(wspace=0, hspace=0)
#plt.title('goodPSF selected top 25 stars', loc='left')
plt.show()


# go pull this directly from new cutout file
xs_best = np.array(xs_best)
ys_best = np.array(ys_best)

starChooser=psfStarChooser.starChooser(img_data,
                                            xs_best,ys_best,
                                            xs_best*500,xs_best*1.0)

(goodFits, goodMeds, goodSTDs) = starChooser(30,200,noVisualSelection=True,autoTrim=False,
                                            bgRadius=15, quickFit = False,
                                            printStarInfo = True,
                                            repFact = 5, ftol=1.49012e-08)

NN_top25_PSF = psf.modelPSF(np.arange(61),np.arange(61), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
NN_top25_PSF.genLookupTable(img_data, goodFits[:,4], goodFits[:,5], verbose=False)


xs_threshold = np.array(xs_threshold)
ys_threshold = np.array(ys_threshold)

starChooser=psfStarChooser.starChooser(img_data,
                                            xs_threshold, ys_threshold,
                                            xs_threshold*500,xs_threshold*1.0)

(goodFits, goodMeds, goodSTDs) = starChooser(30,200,noVisualSelection=True,autoTrim=False,
                                            bgRadius=15, quickFit = False,
                                            printStarInfo = True,
                                            repFact = 5, ftol=1.49012e-08)

NN_threshold_PSF = psf.modelPSF(np.arange(61),np.arange(61), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
NN_threshold_PSF.genLookupTable(img_data, goodFits[:,4], goodFits[:,5], verbose=False)




# make fig with both of these
figure, axes = plt.subplots(nrows=1, ncols=3, figsize = (10,8))
#plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
(z1, z2) = zscale.get_limits(NN_threshold_PSF.lookupTable)
normer = interval.ManualInterval(z1,z2)
axes[1].imshow(normer(NN_threshold_PSF.lookupTable))
title0 = 'ZScaled ' + inputFile.replace('.fits','.NN_threshold_PSF.fits') #right titles?
axes[1].set_title(title0,fontsize=12)
#plt.gca().axes.get_xaxis().set_visible(False)
#plt.gca().axes.get_yaxis().set_visible(False)


(z1, z2) = zscale.get_limits(NN_top25_PSF.lookupTable)
normer = interval.ManualInterval(z1,z2)
axes[0].imshow(normer(NN_top25_PSF.lookupTable))
title1 = 'ZScaled ' + inputFile.replace('.fits','.NN_top25_PSF.fits') #right titles?
axes[0].set_title(title0,fontsize=12)


otherPSF = psf.modelPSF(restore=comparePSF)
(o1, o2) = zscale.get_limits(otherPSF.lookupTable)
normer2 = interval.ManualInterval(o1,o2)
axes[2].imshow(normer2(otherPSF.lookupTable))
title1 = 'ZScaled ' + inputFile.replace('.fits','.goodPSF.fits')
axes[2].set_title(title1,fontsize=12)


'''
axs[1].set_xticks([])
axs[1].set_yticks([])

# X-axis tick label
plt.xticks(color='w')
# Y-axis tick label
plt.yticks(color='w')

frame2 = plt.gca()
for xlabel_i in frame2.axes.get_xticklabels():
    xlabel_i.set_visible(False)
    xlabel_i.set_fontsize(0.0)
for xlabel_i in frame2.axes.get_yticklabels():
    xlabel_i.set_fontsize(0.0)
    xlabel_i.set_visible(False)
for tick in frame2.axes.get_xticklines():
    tick.set_visible(False)
for tick in frame2.axes.get_yticklines():
    tick.set_visible(False)
plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)
'''
plt.show()