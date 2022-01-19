import json
import matplotlib.pyplot as plt
from HSCgetStars_func import HSCgetStars_main 
from HSCpolishPSF_func import HSCpolishPSF_main
import os

def get_user_input():
    '''
    '''
    rewrite_cutouts_str = input("Rewrite cutouts (Y/N): ")
    if rewrite_cutouts_str == 'Y':
        rewrite_cutouts = True 
    else: 
        rewrite_cutouts = False
    fixed_size_str = input("Use fixed cutout size?") # communicate this better
    fixed_cutout_len = int(input("Only cuotus of x length or fixed? (Y/N): ")) #tune this more
    night_dir = input("Night directory (eg. 03074)")
    start_indx = int(input("Image start index (eg. 219502): "))
    end_indx = int(input("Image end index? (eg. 219620): "))

    return fixed_cutout_len, rewrite_cutouts, night_dir, start_indx, end_indx

def int_to_str(i):
    '''
    '''
    if i<10:
        num_str = '00' + str(i)
    elif i<100:
        num_str = '0' + str(i)
    elif i>100:
        num_str = str(i)
    else:
        print('ERROR')
        break 
    return num_str

def main():

    val = input("Change default values (Y/N): ")
    if val == 'Y':

        fixed_cutout_len, rewrite_cutouts, night_dir, start_indx, end_indx = get_user_input()

    else:
        fixed_cutout_len = 111 
        rewrite_cutouts = False 
        night_dir = '03074'
        start_indx = 219580 
        end_indx = 219620


    for k in range(start_indx, end_indx, 2):
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

            num_str = int_to_str(i)

            file_in = 'CORR-0' + str(k) + '-' + num_str + '.fits'
            file_psf = 'psfStars/CORR-0' + str(k) + '-' + num_str + '.psf_cleaned.fits'

            outFile = file_dir + '/' + file_in.replace('.fits', + str(fixed_cutout_len)  +'_cutouts_savedFits.pickle')

            
            # if PSF already exists, and just interested in top 25, is it more efficient just to grab from header?
            if os.path.isfile(outFile):
                print('HSCgetStars already successfully run, skipping to HSCpolishPSF')
            else: 
                HSCgetStars_main(fixed_cutout_len=fixed_cutout_len, dir = file_dir, inputFile = file_in, psfFile = file_psf)

            HSCpolishPSF_main(fixed_cutout_len=fixed_cutout_len, dir=file_dir, inputFile=file_in, cutout_file=outFile)


if __name__ == '__main__':
    main()