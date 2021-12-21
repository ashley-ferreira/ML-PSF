import json
import matplotlib.pyplot as plt
from HSCgetStars_func import HSCgetStars_main 
from HSCpolishPSF_func import HSCpolishPSF_main
import os

invalid_files = [216676]

for k in range(216700, 216814,2):  #May26 216652 216814, wrong numbers?
    print(k)
    if k in invalid_files:
        print('not including')
        continue 
    
    file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03068/HSC-R2/corr' # can generalize $USER in future

    for i in range(0,103):
        if i == 9:
            print('chip 9 broken, not including')
            continue 
        elif i == 32:
            print('chip 32 half broken, not including')
            continue
        elif i == 19:
            print('skipping chip 19')
            continue

        if i<10:
            num_str = '00' + str(i)
        elif i<100:
            num_str = '0' + str(i)
        elif i>100:
            num_str = str(i)
        else:
            print('ERROR')
            break 

        file_in = 'CORR-0' + str(k) + '-' + num_str + '.fits'
        file_psf = 'psfStars/CORR-0' + str(k) + '-' + num_str + '.psf_cleaned.fits'

        outFile = file_dir + '/' + file_in.replace('.fits', '_cutouts_savedFits.pickle')

        
        # if PSF already exists, and just interested in top 25, is it more efficient just to grab from header?
        if os.path.isfile(outFile):
            print('HSCgetStars already successfully run, skipping to HSCpolishPSF')
        else: 
            HSCgetStars_main(dir = file_dir, inputFile = file_in, psfFile = file_psf)

        HSCpolishPSF_main(dir=file_dir, inputFile=file_in)