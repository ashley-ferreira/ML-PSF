from HSCgetStars_func import HSCgetStars_main 
from HSCpolishPSF_func import HSCpolishPSF_main
import os
import sys

def get_user_input():
    '''
    Prompts user for data prep parameters/options

    Parameters:    

        None

    Returns:
        
        fixed_cutout_len (int): force cutouts to have shape
                                (fixed_cutout_len, fixed_cutout_len)
              --> set to zero for cutoutWidth = max(30, int(5*fwhm))
        
        rewrite_cutouts (bool): if cutout_file already exists
                                --> True = rewrite file
                                --> False = don't rewrite
        
        night_dir (str): directory which corresponds to night of interest
        
        start_indx (int): first image for range of interest
        
        end_indx (int): last image for range of interest
        
    '''
    rewrite_cutouts_str = input("Rewrite cutouts (Y/N):")
    if rewrite_cutouts_str == 'Y':
        rewrite_cutouts = True 
    else: 
        rewrite_cutouts = False

    fixed_size_str = input("Enter specific cutout size or 0 for FWHM determined size:") 
    night_dir = input("Night directory (eg. 03074):")
    start_indx = int(input("Image start index (eg. 219502):"))
    end_indx = int(input("Image end index? (eg. 219620):"))

    return fixed_cutout_len, rewrite_cutouts, night_dir, start_indx, end_indx

def int_to_str(i):
    '''
    Takes in integer i and output three digit string version using 
    zeros on the left to pad if number is less than 3 digits

    Parameters:    

        i (int): integer value between 0 and 999 to convert to string

    Returns:
        
        num_str (str): three digit string version of i

    '''    

    if i<10:
        num_str = '00' + str(i)
    elif i<100:
        num_str = '0' + str(i)
    elif i>100:
        num_str = str(i)
    else:
        print('ERROR')
        sys.exit()

    return num_str

def main():

    val = input("Change default values (Y/N): ")
    if val == 'Y':
        fixed_cutout_len, rewrite_cutouts, night_dir, start_indx, end_indx = get_user_input()
    else:
        fixed_cutout_len = 111 
        rewrite_cutouts = False 
        night_dir = '03068'
        start_indx = 216730
        end_indx = 216732

    # loop over all images in range (only even numbers exist)
    for k in range(start_indx, end_indx, 2):
        
        # can generalize $USER in future
        file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03068/HSC-R2/corr' 

        # loop over all CCD chips
        for i in range(0,103):
            if i == 9:
                print('chip 9 broken, not including')
                continue 
            elif i == 32:
                print('chip 32 half broken, not including')
                continue 

            # define input file
            num_str = int_to_str(i)
            file_in = 'CORR-0' + str(k) + '-' + num_str + '.fits'

            try:
                # check if HSCgetStars_main has already been run (AKA if cutout file exists)
                cutout_file = file_dir + '/' + file_in.replace('.fits', str(fixed_cutout_len) \
                                                            + '_cutouts_savedFits.pickle')
                if os.path.isfile(cutout_file) and rewrite_cutouts = False:
                    print('HSCgetStars already successfully run, skipping to HSCpolishPSF')
                else: 
                    HSCgetStars_main(file_dir, file_in, cutout_file, fixed_cutout_len)

                # run HSCpolishPSF_main no matter what
                HSCpolishPSF_main(file_dir, file_in, cutout_file, fixed_cutout_len)

            except Exception as e: 
                print(e)    

if __name__ == '__main__':
    main()