import pylab as pyl
from trippy import psf, psfStarChooser
import os
import sys
from astropy.visualization import interval, ZScaleInterval
from astropy.io import fits
import math 
import numpy as np
import matplotlib.gridspec as gridspec
import pickle
import keras
import matplotlib.pyplot as plt
from optparse import OptionParser
parser = OptionParser()
zscale = ZScaleInterval()

pwd = '/arc/projects/uvickbos/ML-PSF/'

model_dir = pwd + 'Saved_Model/' + '2022-04-23-13:53:44/models_lesslay16_256_lr=0.001_drop=0.2_split=0.2/'

cutout_size = 111

night_dir = '03074'

parser.add_option('-C', '--conf_cutoff', dest='conf_cutoff', 
        default='0.95', type='float', \
        help='confidence cutoff, default=%default.')

parser.add_option('-S', '--SNR_proxy_cutoff', dest='SNR_proxy_cutoff', 
        default='10.0', type='float', 
        help='SNR proxy cutoff, default=%default.')

parser.add_option('-s', '--min_num_stars', dest='min_num_stars', 
        default='10', type='int', 
        help='minimum number of stars acceptable, default=%default.')

parser.add_option('-f', '--file_dir', dest='file_dir', 
    default=pwd+'home_dir_transfer/HSC_May25-lsst/rerun/processCcdOutputs/'+night_dir+'/HSC-R2/corr/', 
    type='str', help='directory which contains data, default=%default.')

default_data_dir = pwd+'/NN_data_' + str(cutout_size) + '/'
parser.add_option('-d', '--data_dir', dest='data_dir', 
    default=default_data_dir, type='str', 
    help='directory where cutouts are saved, default=%default.')

def crop_center(img, cropx, cropy):
    '''
    Crops input image array around center to desired (cropx, cropx) size
    
    Taken from stack overflow: 
    https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image

    Parameters:    

        img (arr): image to be cropped

        cropx (int): full width of desired cutout

        cropy (int): full height of desired cutout

    Returns:
        
        cropped_img (arr): cropped image

    '''
    x,y = img.shape 
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)
    cropped_img = img[int(startx):int(startx+cropx), int(starty):int(starty+cropy)]

    return cropped_img

def regularize(cutouts, mean, std):
    '''
    Regularizes either single cutout or array of cutouts

    Parameters:

        cutouts (arr): cutouts to be regularized

        mean (float): mean used in training data regularization  

        std (float): std used in training data regularization

    Returns:

        regularized_cutout (arr): regularized cutout
    
    '''
    cutouts = np.asarray(cutouts).astype('float32')
    cutouts -= mean
    cutouts /= std
    w_bad = np.where(np.isnan(cutouts))
    cutouts[w_bad] = 0.0
    regularized_cutout = cutouts

    return regularized_cutout

def get_user_input():
    '''
    Prompts user for inputs

    Parameters:    
    
        None 

    Returns: 

        input_file (str): file of interest to generate PSF for
        
        file_dir (str): directory where input_file is saved

        data_dir (str): directory where source cutouts from input_file are saved
        
        model_dir_name (str): directory where Neural Network model is saved
        
        NN_cutoff_vals (lst): cutoff values that Neural Network uses when picking
                              best stars, [conf_cutoff, SNR_proxy_cutoff, min_num_stars]

        cutout_size (int): defines shape of cutouts (cutout_size, cutout_size)

    '''
    (options, args) = parser.parse_args()

    model_dir_name = model_dir 

    NN_cutoff_vals = [options.conf_cutoff, options.SNR_proxy_cutoff, options.min_num_stars]

    # try and use this below instead in future
    #file_dir = options.pwd + 'HSC_May25-lsst/rerun/processCcdOutputs/' + options.night_dir + '/HSC-R2/corr/'
    file_dir = options.file_dir

    return file_dir, options.data_dir, model_dir_name, NN_cutoff_vals, options.cutout_size


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
    elif i>=100:
        num_str = str(i)
    else:
        print('ERROR: incorrect i=',i)
        sys.exit()

    return num_str

    
def NN_PSF_generate(inputs, input_file):
    '''
    

    Parameters: 

        input_file (str): file of interest to generate PSF for
        
        file_dir (str): directory where input_file is saved

        data_dir (str): directory where source cutouts from input_file are saved
        
        model_dir_name (str): directory where Neural Network model is saved
        
        NN_cutoff_vals (lst): cutoff values that Neural Network uses when picking
                              best stars, [conf_cutoff, SNR_proxy_cutoff, min_num_stars]

        cutout_size (int): defines shape of cutouts (cutout_size, cutout_size)

    Returns:

        None 

    '''
    # unpack inputs
    file_dir, data_dir, model_dir_name, NN_cutoff_vals, cutout_size = inputs

    # unpack cutoff values
    conf_cutoff = NN_cutoff_vals[0]
    SNR_proxy_cutoff = NN_cutoff_vals[1]
    min_num_stars = NN_cutoff_vals[2]
        
    # read in cutout data for input_file
    outFile_wMetadata = data_dir+input_file.replace('.fits', '_'+str(cutout_size)+'_cutouts_savedFits.pickle')

    if os.path.exists(outFile_wMetadata):
        with open(outFile_wMetadata, 'rb') as han:
            [std, seconds, peaks, xs, ys, cutouts, fwhm, inputFile] = pickle.load(han)

    else:
        print('could not find cutout file, tried:')
        print(outFile_wMetadata)

    # load previously trained Neural Network 
    model = keras.models.load_model(model_dir_name)
    '''
    model_found = False 
    for file in os.listdir(model_dir_name):
        if file.startswith('model_traintime=*'):
            model = keras.models.load_model(model_dir_name + file)
            model_found = True
            break
    if model_found == False: 
        print('ERROR: no model file in', model_dir_name)
        sys.exit()
    '''

    # load training set std and mean
    # TEMP WAY
    reg_dir = '/arc/projects/uvickbos/ML-PSF/' + 'Saved_Model/' + '2022-04-23-13:53:44/'
    with open(reg_dir + 'regularization_data.pickle', 'rb') as han:
        [std, mean] = pickle.load(han)

    # use std and mean to regularize cutout
    cutouts = regularize(cutouts, mean, std)

    xs_best = []
    ys_best = []
    cn_prob = []

    output = model.predict(cutouts)
    for i in range(len(cutouts)):
        good_probability = output[i][1]
        cn_prob.append(good_probability) 

    cn_prob, xs, ys, cutouts = zip(*sorted(zip(cn_prob, xs, ys, cutouts), reverse = True))

    for i in range(len(cutouts)): 
        if plotted_stars < 25:
            good_probability = cn_prob[i]
            #center = crop_center(cutouts[i],5,5)
            #sum_c = center.sum()
            #SNR_proxy = math.sqrt(abs(sum_c))
            print(good_probability)#, SNR_proxy)
            if good_probability > conf_cutoff:  #SNR_proxy > SNR_proxy_cutoff and      
                xs_best.append(xs[i])
                ys_best.append(ys[i])
                plotted_stars += 1 

    if plotted_stars < min_num_stars: 
        print('You requested a minimum of', min_num_stars)
        print('However there are only', plotted_stars, 'with confidence >', \
            conf_cutoff, 'and SNR proxy >', SNR_proxy_cutoff)
        print('Please lower one of these numbers and try again')
        sys.exit()

    # load image data
    with fits.open(file_dir+input_file) as han:
        img_data = han[1].data.astype('float64')
        img_header = han[0].header

    xs_best = np.array(xs_best)
    ys_best = np.array(ys_best)


    # read in saved fileinstead!
    starChooser=psfStarChooser.starChooser(img_data,
                                                xs_best,ys_best,
                                                xs_best*500,xs_best*1.0)

    (goodFits, goodMeds, goodSTDs) = starChooser(30,200,noVisualSelection=True,autoTrim=False,
                                                bgRadius=15, quickFit = False,
                                                printStarInfo = True,
                                                repFact = 5, ftol=1.49012e-08)

    NN_top25_PSF = psf.modelPSF(np.arange(61),np.arange(61), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
    NN_top25_PSF.genLookupTable(img_data, goodFits[:,4], goodFits[:,5], verbose=False)

    '''
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize = (10,8))
    (z1, z2) = zscale.get_limits(NN_top25_PSF.lookupTable)
    normer = interval.ManualInterval(z1,z2)
    axes[0].imshow(normer(NN_top25_PSF.lookupTable))
    title1 = 'ZScaled ' + input_file.replace('.fits','.NN_PSF.fits') 
    axes[0].set_title(title1,fontsize=12)
    '''


def main():
    for i in range(219610, 219622, 2): # just rough idea for now?
        randos = np.random.choice(range(103), 5, replace=False)
        for r in randos:
            num_str = int_to_str(r)
            file_in = 'CORR-0' + str(i) + '-' + num_str + '.fits'

    NN_PSF_generate(get_user_input(), file_in)
    
if __name__ == '__main__':
    main()