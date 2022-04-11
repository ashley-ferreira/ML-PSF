#from sympy import trailing
from re import A
from HSCgetStars_func import HSCgetStars_main 
from HSCpolishPSF_func import HSCpolishPSF_main
import os
import sys
from optparse import OptionParser
parser = OptionParser()

parser.add_option('-r', '--retwrite_cutouts', dest='rewrite_cutouts', 
    default='0', type='int',  
    help='"1" to retwrite cutouts, "0" to not, default=%default.')

fixed_cutout_len = 111
parser.add_option('-l', '--fixed_cutout_length', dest='fixed_cutout_len', 
    default=fixed_cutout_len, type='int', 
    help='l is size of cutout required, produces (l,l) shape. enter 0 for FWHM*5 size default=%default.')

night_dir = '03074'
parser.add_option('-n', '--night_dir', dest='night_dir', 
    default=night_dir, type='str', help='directory for specific night to use, default=%default.')

parser.add_option('-f', '--file_dir', dest='file_dir', 
    default='/arc/projects/uvickbos/ML-PSF/home_dir_transfer/HSC_May25-lsst/rerun/processCcdOutputs/'+night_dir+'/HSC-R2/corr/', 
    type='str', help='directory which contains data, default=%default.')
    
parser.add_option('-s', '--start_indx', dest='start_indx', 
    default='219502', type='int', help='index of image to start on, default=%default.')

parser.add_option('-e', '--end_indx', dest='end_indx', 
    default='219620', type='int', help='index of image to end on, default=%default.')

default_training_dir = '/arc/projects/uvickbos/ML-PSF/NN_data_' + str(fixed_cutout_len) + '/'
parser.add_option('-t', '--training_dir', dest='training_dir', 
    default=default_training_dir, type='str', 
    help='directory to save cutouts to for training, default=%default.')


def get_user_input(): 
    '''
    Gets user preferences for data prep parameters/options

    Parameters:    

        None

    Returns:

        file_dir (str): directory to access data
        
        fixed_cutout_len (int): force cutouts to have shape
                                (fixed_cutout_len, fixed_cutout_len)
              --> set to zero for cutoutWidth = max(30, int(5*fwhm))
        
        rewrite_cutouts (bool): if cutout_file already exists
                                --> 1 = rewrite file
                                --> 0 = don't rewrite
        
        night_dir (str): directory which corresponds to night of interest
        
        start_indx (int): first image for range of interest
        
        end_indx (int): last image for range of interest
        
    '''

    (options, args) = parser.parse_args()

    return options.file_dir, options.fixed_cutout_len, options.rewrite_cutouts, \
         options.night_dir, options.start_indx, options.end_indx, options.training_dir

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
    ''' 
    This program creates the data needed to perform Star_NN_dev.py training on 
    '''

    file_dir, fixed_cutout_len, rewrite_cutouts, night_dir, start_indx, end_indx, training_dir = get_user_input()

    # loop over all images in range (only even numbers exist)
    for k in range(start_indx, end_indx, 2):

        # loop over all CCD chips
        for i in range(0,103):
            # define input file
            num_str = int_to_str(i)
            file_in = 'CORR-0' + str(k) + '-' + num_str + '.fits'
            
            if i == 9:
                print('chip 9 broken, not including')
                continue 

            elif os.path.isfile(file_dir + file_in):

                if True:#try:
                    # check if HSCgetStars_main has already been run (AKA if cutout file exists)
                    cutout_file = file_dir + '/' + file_in.replace('.fits', str(fixed_cutout_len) 
                                                                + '_cutouts_savedFits.pickle')
                    if os.path.isfile(cutout_file) and rewrite_cutouts == 0:
                        print('HSCgetStars already successfully run, skipping to HSCpolishPSF')
                    else: 
                        HSCgetStars_main(file_dir, file_in, cutout_file, fixed_cutout_len)

                    # run HSCpolishPSF_main no matter what
                    HSCpolishPSF_main(file_dir, file_in, cutout_file, fixed_cutout_len, training_dir)
                '''
                except Exception as Argument:
                    print(Argument)

                    # creating/opening a file
                    err_log = open(training_dir + 'data_prep_error_log.txt', 'a')

                    # writing in the file
                    err_log.write('DataPrepLoop.py' + file_in + str(Argument))
                    
                    # closing the file
                    err_log.close()   
                '''
            else: 
                print(file_dir + file_in, ' does not exist')

if __name__ == '__main__':
    main()