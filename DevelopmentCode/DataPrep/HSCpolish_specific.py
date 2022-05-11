from re import A
from HSCgetStars_func import HSCgetStars_main 
from HSCpolishPSF_func import HSCpolishPSF_main
import os
import sys

file_in = 'CORR-0219612-077.fits'
training_dir = '/arc/projects/uvickbos/ML-PSF/NN_data_111/'
file_dir = '/arc/projects/uvickbos/ML-PSF/home_dir_transfer/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr/'

cutout_file = training_dir + file_in.replace('.fits', '_' + str(111) 
                                                                + '_cutouts_savedFits.pickle')

HSCpolishPSF_main(file_dir, file_in, cutout_file, 111, training_dir)