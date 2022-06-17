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
parser.add_option('-p', '--pwd', dest='pwd', 
        default=pwd, type='str', 
        help=', default=%default.')

model_dir = pwd + 'Saved_Model/' 
parser.add_option('-m', '--model_dir_name', dest='model_name', \
        default='default_model/', type='str', \
        help='name for model directory, default=%default.')

cutout_size = 111
parser.add_option('-c', '--cutout_size', dest='cutout_size', \
        default=cutout_size, type='int', \
        help='c is size of cutout required, produces (c,c) shape, default=%default.')

night_dir = '03068' #'03074'
parser.add_option('-n', '--night_dir', dest='night_dir', 
        default=night_dir, type='str', \
        help='image file directory, default=%default.')

parser.add_option('-i', '--img_file', dest='img_file', 
        default='0216652-000', type='str', \
        help='input file to use for comparison, default=%default.')

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

default_data_dir = pwd+'/NN_data_' + str(cutout_size) + '_diffnight/'
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

    model_dir_name = model_dir + options.model_name

    NN_cutoff_vals = [options.conf_cutoff, options.SNR_proxy_cutoff, options.min_num_stars]

    input_file = 'CORR-' + str(options.img_file) + '.fits'

    #file_dir = options.pwd + 'HSC_May25-lsst/rerun/processCcdOutputs/' + options.night_dir + '/HSC-R2/corr/'
    file_dir = options.file_dir

    return input_file, file_dir, options.data_dir, model_dir_name, NN_cutoff_vals, options.cutout_size

    
def compare_NN_goodPSF(inputs):
    '''
    Compares top 25 stars chosen from goodPSF method to top stars chosen by 
    Neural network by plotting them next to one another. Creates and pltos two 
    PSFs, one from goodPSF top 25 stars and one from Neural Network chosen stars.

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
    input_file, file_dir, data_dir, model_dir_name, NN_cutoff_vals, cutout_size = inputs

    # unpack cutoff values
    conf_cutoff = NN_cutoff_vals[0]
    SNR_proxy_cutoff = NN_cutoff_vals[1]
    min_num_stars = NN_cutoff_vals[2]
        
    # read in cutout data for input_file
    outFile_wMetadata = data_dir+input_file.replace('.fits', '_'+str(cutout_size)+'_cutouts_savedFits.pickle')
    #outFile_simple = file_dir+input_file.replace('.fits', '_cutouts_savedFits.pickle')
    if os.path.exists(outFile_wMetadata):
        with open(outFile_wMetadata, 'rb') as han:
            [std, seconds, peaks, xs, ys, cutouts, fwhm, inputFile] = pickle.load(han)
    #elif os.path.exists(outFile_simple):
    #    with open(outFile_simple, 'rb') as han:
    #        [std, seconds, peaks, xs, ys, cutouts] = pickle.load(han)
    else:
        print('could not find cutout file, tried:')
        print(outFile_wMetadata)
        #print(outFile_simple)

    cutouts_cleaned = []
    for cutout in cutouts: # does this need to be forced?
        inf_or_nan = np.isfinite(cutout)
        if False in inf_or_nan:#not(inf_or_nan.any()): # if any is false then go for this
            print("non finite values in array")
            #pass
            # dont need both of these and maybe not either if you have above
        elif cutout.min() < -2000 or cutout.max() > 130000:
            print("very big or small pix vals")
            #pass
        #elif cutout.min() < -200 or cutout.max() > 65536:
        elif cutout.min() < -200 or cutout.max() > 65536/2:
            print("big or small pix value")
            #label = 0 # add this in somehow, take it out
            #pass
        else:
            cutouts_cleaned.append(cutout)

        

            
        
    cutouts_cleaned = np.array(cutouts_cleaned)

    # load previously trained Neural Network 
    model_found = False 
    for file in os.listdir(model_dir_name):
        # TEMP 
        print(file)
        if file.startswith('model_20'): 
            print(file)
            model = keras.models.load_model(model_dir_name + file)
            break 
    '''
        if file.startswith('model_traintime=*'):
            model = keras.models.load_model(model_dir_name + file)
            model_found = True
            break
    if model_found == False: 
        print('ERROR: no model file in', model_dir_name)
        sys.exit()
    '''

    # load training set std and mean
    # TEMP UP ONE FOLDER
    with open(model_dir_name + '../regularization_data.pickle', 'rb') as han:
        [std, mean] = pickle.load(han)

    # use std and mean to regularize cutout
    cutouts = regularize(cutouts_cleaned, mean, std)

    xs_best = []
    ys_best = []
    cn_prob = []

    output = model.predict(cutouts)
    for i in range(len(cutouts)):
        good_probability = output[i][1]
        cn_prob.append(good_probability) 

    cn_prob, xs, ys, cutouts = zip(*sorted(zip(cn_prob, xs, ys, cutouts), reverse = True))

    fig, axs = plt.subplots(5,5,figsize=(10, 12))
    axs = axs.ravel()
    plt.title('NN selected top 25 stars:' + inputFile, x=-1.7, y=6.5) 
    plotted_stars = 0
    for i in range(len(cutouts)): 
        if plotted_stars < 25:#min_num_stars:
            good_probability = cn_prob[i]
            center = crop_center(cutouts[i],5,5)
            sum_c = center.sum()
            SNR_proxy = math.sqrt(abs(sum_c))
            print(good_probability, SNR_proxy)
            if good_probability > conf_cutoff: #SNR_proxy > SNR_proxy_cutoff and       
                xs_best.append(xs[i])
                ys_best.append(ys[i])
                print(good_probability)
                (z1, z2) = zscale.get_limits(cutouts[i])
                normer = interval.ManualInterval(z1,z2)
                axs[plotted_stars].imshow(normer(cutouts[i]))
                axs[plotted_stars].set_xticks([])
                axs[plotted_stars].set_yticks([])
                axs[plotted_stars].text(0.1, -1, 'conf:' + str(good_probability))
                axs[plotted_stars].text(0.1, -15, 'min, max:' + str(cutouts[i].min()) + '   ' + str(cutouts[i].max())[:7])

                plotted_stars += 1 

    if plotted_stars < min_num_stars: 
        print('You requested a minimum of', min_num_stars)
        print('However there are only', plotted_stars, 'with confidence >', \
            conf_cutoff, 'and SNR proxy >', SNR_proxy_cutoff)
        print('Please lower one of these numbers and try again')
        sys.exit()

    plt.subplots_adjust(wspace=0., hspace=0.3)
    plt.show()

    # load image data
    with fits.open(file_dir+input_file) as han:
        img_data = han[1].data.astype('float64')
        img_header = han[0].header
    
    # load goodPSF
    goodPSF = file_dir+'psfStars/'+input_file.replace('.fits','._goodPSF.fits')
    with fits.open(goodPSF) as han:
        goodPSF_img_data = han[1].data
        goodPSF_header = han[0].header

    goodPSF_x = []
    goodPSF_y = []
    count = 0
    for e in goodPSF_header: 
        if e[:5] == 'XSTAR':
            goodPSF_x.append(goodPSF_header[count])
        elif e[:5] =='YSTAR':
            goodPSF_y.append(goodPSF_header[count])
        count += 1

    fig, axs = plt.subplots(5,5,figsize=(5,5))
    axs = axs.ravel()
    plt.title('goodPSF selected top 25 stars:' + input_file, x=-1.5, y=5)
    cutoutWidth = cutout_size // 2
    for i in range(len(goodPSF_x)):
        y_int = int(goodPSF_y[i])
        x_int = int(goodPSF_x[i])
        cutout_goodPSF = img_data[y_int-cutoutWidth:y_int+cutoutWidth+1, x_int-cutoutWidth:x_int+cutoutWidth+1]
   
        # this isnt actually needed?
        cutout_goodPSF = regularize(cutout_goodPSF, mean, std)

        (z1, z2) = zscale.get_limits(cutout_goodPSF)
        normer = interval.ManualInterval(z1,z2)
        axs[i].imshow(normer(cutout_goodPSF))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    
    
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

    
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize = (10,8))
    
    (z1, z2) = zscale.get_limits(NN_top25_PSF.lookupTable)
    normer = interval.ManualInterval(z1,z2)
    axes[0].imshow(normer(NN_top25_PSF.lookupTable))
    title1 = 'ZScaled ' + input_file.replace('.fits','.NN_PSF.fits') 
    axes[0].set_title(title1,fontsize=12)
    
    
    restored_goodPSF = psf.modelPSF(restore=goodPSF)
    (z1, z2) = zscale.get_limits(restored_goodPSF.lookupTable)
    normer = interval.ManualInterval(z1,z2)
    axes[1].imshow(normer(restored_goodPSF.lookupTable))
    title2 = 'ZScaled ' + input_file.replace('.fits','.goodPSF.fits')
    axes[1].set_title(title2,fontsize=12)

    plt.show()
    
def main():
    compare_NN_goodPSF(get_user_input())
    
if __name__ == '__main__':
    main()