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

file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr'

size_of_data = int(sys.argv[1])//2 # try small set just to debug

good_cutouts_size = [] # label 1
bad_cutouts_size = [] # label 0
all_cutouts_size = [] 

# can do this on an image specific way later too, doc the fits, fwhm too

class BreakException(Exception):
        pass

files_counted = 0
try:
    for filename in os.listdir(file_dir+ '/NN_data_n=25'):
        if filename.endswith("_cutoutData.pickle"):
            print(files_counted, '/', size_of_data) #fix formatting
            if files_counted >= size_of_data:
                raise BreakException
            print(files_counted, size_of_data)
            #print('file being processed: ', filename)

            with open(file_dir + '/NN_data_n=25/' + filename, 'rb') as f:
                [n, cutout, label] = pickle.load(f)
            
            size = len(cutout)
            all_cutouts_size.append(size)

            if label == 1:
                good_cutouts_size.append(size)
                files_counted += 1
            elif label == 0:
                bad_cutouts_size.append(size)
                #files_counted += 1
            else:
                print('ERROR: label is not 1 or 0')

    
except BreakException:
    pass

with open(file_dir + '/NN_data_n=25_histogram/cutout_hist.pickle', 'wb+') as han:
    pickle.dump([all_cutouts_size, good_cutouts_size, bad_cutouts_size, bad_cutouts_size], han)

# make bins same for all
bins=range(min(all_cutouts_size), max(all_cutouts_size) + 1, 1)
pyl.hist(all_cutouts_size, bins=bins, label='all cutouts')#, rwidth=0.1)
pyl.hist(bad_cutouts_size, bins=bins, label='bad cutouts')#, rwidth=0.1)
pyl.hist(good_cutouts_size, bins=bins, label='good cutouts')#, rwidth=0.1)
pyl.ylabel('Count')
pyl.xlabel('Cutout Size')
pyl.legend(loc='best')
pyl.show() 

pyl.hist(good_cutouts_size, bins='auto', label='good cutouts', rwidth=0.5)
pyl.ylabel('Count')
pyl.xlabel('Cutout Size')
pyl.legend(loc='best')
pyl.show()