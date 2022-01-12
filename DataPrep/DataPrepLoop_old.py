import json
import matplotlib.pyplot as plt
from HSCgetStars_func import HSCgetStars_main 
from HSCpolishPSF_func import HSCpolishPSF_main
import os

fixed_cutout_len = 111

for k in range(219502,219620,2): 
    print(k)
    
    file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr' # can generalize $USER in future

    for i in range(0,103):
        if i == 9:
            print('chip 9 broken, not including')
            continue 
        elif i == 32:
            print('chip 32 half broken, not including')
            continue
        elif i == 19:
            print('skipping chip 19') #use try thing instead
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

        outFile = file_dir + '/' + file_in.replace('.fits', + str(fixed_cutout_len)  +'_cutouts_savedFits.pickle')

        
        # if PSF already exists, and just interested in top 25, is it more efficient just to grab from header?
        if os.path.isfile(outFile):
            print('HSCgetStars already successfully run, skipping to HSCpolishPSF')
        else: 
            HSCgetStars_main(fixed_cutout_len=fixed_cutout_len, dir = file_dir, inputFile = file_in, psfFile = file_psf)

        HSCpolishPSF_main(fixed_cutout_len=fixed_cutout_len, dir=file_dir, inputFile=file_in, cutout_file=outFile)