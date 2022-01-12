import json
import matplotlib.pyplot as plt
from HSCgetStars_func import HSCgetStars_main 
from HSCpolishPSF_func import HSCpolishPSF_main
import os

#invalid_files = [216676]

start_indx = 219582
end_indx = 219620 


for k in range(start_indx, end_indx, 2):
    print(k)

    try:
        #if k in invalid_files:
        #    print('not including')
        #    continue 
        
        file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03074/HSC-R2/corr' # can generalize $USER in future

        for i in range(0,103):
            if i == 9:
                print('chip 9 broken, not including')
                continue 
            elif i == 32:
                print('chip 32 half broken, not including')
                continue
            #elif i == 19:
            #    print('skipping chip 19')
            #    continue
            #elif i == 70 and k == 216700:
            #    continue 

            if i<10:
                num_str = '00' + str(i)
            elif i<100:
                num_str = '0' + str(i)
            elif i>100:
                num_str = str(i)
            #else:
            #    print('ERROR')
            #    break 

            file_in = 'CORR-0' + str(k) + '-' + num_str + '.fits'
            file_psf = 'psfStars/CORR-0' + str(k) + '-' + num_str + '.psf_cleaned.fits'

            #if rewrite_cutouts == True and 
            
            HSCgetStars_main(fixed_cutout_len = 111, dir = file_dir, inputFile = file_in, psfFile = file_psf)

    except Exception as e: 
        print('FAILURE')
        print(e)