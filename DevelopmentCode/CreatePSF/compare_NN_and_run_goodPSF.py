import pylab as pyl
from trippy import psf, psfStarChooser
import time
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

import pickle as pick, numpy as np, pylab as pyl
from sklearn import cluster
from astropy.visualization import interval, ZScaleInterval
from trippy import psf, psfStarChooser
from astropy.io import fits
import sys
import time, os
from trippy import psf, scamp, psfStarChooser, bgFinder, MCMCfit

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
        default='0.9', type='float', \
        help='confidence cutoff, default=%default.')

parser.add_option('-S', '--SNR_proxy_cutoff', dest='SNR_proxy_cutoff', 
        default='10.0', type='float', 
        help='SNR proxy cutoff, default=%default.')

parser.add_option('-s', '--min_num_stars', dest='min_num_stars', 
        default='20', type='int', 
        help='minimum number of stars acceptable, default=%default.')

parser.add_option('-f', '--file_dir', dest='file_dir', 
    default=pwd+'home_dir_transfer/HSC_May25-lsst/rerun/processCcdOutputs/'+night_dir+'/HSC-R2/corr/', 
    type='str', help='directory which contains data, default=%default.')

default_data_dir = pwd+'/NN_data_' + str(cutout_size) + '_diffnight/'
parser.add_option('-d', '--data_dir', dest='data_dir', 
    default=default_data_dir, type='str', 
    help='directory where cutouts are saved, default=%default.')

#def HSCpolishPSF_main(file_dir, input_file, cutout_file, fixed_cutout_len, training_dir):
def non_ML_timing(file_dir, input_file, cutout):
    '''

    '''

    ### open the fits image
    with fits.open(file_dir+'/'+input_file) as han:
        img_data = han[1].data.astype('float64')
        header = han[0].header
    (A,B)  = img_data.shape

    ### run sextractor
    scamp.makeParFiles.writeSex('HSC.sex',
                        minArea=3.,
                        threshold=5.,
                        zpt=27.8,
                        aperture=20.,
                        min_radius=2.0, 
                        catalogType='FITS_LDAC',
                        saturate=64000)
    scamp.makeParFiles.writeConv()
    # numAps is thenumber of apertures that you want to use. Here we use 1
    scamp.makeParFiles.writeParam(numAps=1) 

    fits.writeto('junk.fits', img_data, header=header, overwrite=True)
    scamp.runSex('HSC.sex', 'junk.fits' ,options={'CATALOG_NAME':f'{file_dir}/{input_file}.cat'},verbose=False)
 
    print('starting non-ML timer')
    start_noml = time.time()
    # just to get a rough idea of time
    catalog = scamp.getCatalog(f'{file_dir}/{input_file}.cat',paramFile='def.param')
    os.system('rm junk.fits')

    ## select only high SNR stars
    w = np.where(catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO'] > 200)
    X_ALL = catalog['XWIN_IMAGE'][w]
    Y_ALL = catalog['YWIN_IMAGE'][w]

    ## load the PSF that Wes generated at an earlier point. This is not a great PSF!
    file_psf = input_file.replace('.fits','.psf_cleaned.fits')
    goodPSF = psf.modelPSF(restore=file_dir+'psfStars/'+file_psf)
    print(time.time()-start_noml)

    cutoutWidth = 111//2
    # error below now

    ## fit each star with the PSF to get its brightness and position.
    #start = time.time()
    start2_noml = time.time()
    stds, seconds, peaks, xs, ys = [], [], [], [], []
    for i in range(len(X_ALL)):
        x, y = X_ALL[i], Y_ALL[i]
        
        x_int, y_int = int(x), int(y)
        if not (x_int>cutoutWidth+2 and y_int>cutoutWidth+2 and \
                x_int<(B-cutoutWidth-2) and y_int<(A-cutoutWidth-2)):
            continue

        peak = np.max(cutout[cutoutWidth-1:cutoutWidth+2, cutoutWidth-1:cutoutWidth+2])
        peaks.append(peak)

        #background estimate
        bgf = bgFinder.bgFinder(cutout)
        bg_estimate = bgf()

        try:
            fitter = MCMCfit.LSfitter(goodPSF, cutout)
            fitPars = fitter.fitWithModelPSF(cutoutWidth, cutoutWidth, m_in=10.0,
                                            fitWidth = 7, ftol = 1.49e-6, verbose = False)
            if fitPars[2]<=0:
                fitPars=None
        except:
            fitPars = None

        if fitPars  is not None:

            (aa,bb) = cutout.shape
            if int(aa) == 1+(cutoutWidth*2) and int(bb) == 1+(cutoutWidth*2):

                model_cutout = goodPSF.plant(fitPars[0], fitPars[1], fitPars[2], 
                                        cutout*0.0,returnModel=True,addNoise=False)
                pixel_weights = 1.0/(np.abs(model_cutout)+1.0)
                pixel_weights /= np.sum(pixel_weights)

                rem_cutout = goodPSF.remove(fitPars[0], fitPars[1], fitPars[2], 
                                                        cutout,useLinePSF=False)

                weighted_std = np.sum(pixel_weights*(rem_cutout - np.mean(rem_cutout))**2)/np.sum(model_cutout)*(aa*bb)

                sorted = np.sort(rem_cutout.reshape(aa*bb))
                second_highest = sorted[-2]

                stds.append(weighted_std)
                seconds.append(second_highest)
                xs.append(x)
                ys.append(y)

    # store everything in numpy array format
    stds = np.array(stds)
    seconds = np.array(seconds)
    xs = np.array(xs)
    ys = np.array(ys)
    peaks = np.array(peaks)
    
    ## select only those stars with really low STD
    w = np.where(stds/np.std(stds)<0.001)
    stds = stds[w]
    seconds = seconds[w]
    peaks = peaks[w]
    xs = xs[w]
    ys = ys[w]
    s = np.std(stds)

    ## find the best 25 stars (the closest to the origin in 
    ## weighted STD and second highest pixel value)
    dist = ((stds/s)**2 + (seconds/peaks)**2)**0.5
    args = np.argsort(dist)
    best = args[:25]

    end_noml = time.time()
    print('non-ML process completed in', round(end_noml-start2_noml, 10), ' seconds')
    # this is much shorter than what is timed?

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

    non_ML_timing(file_dir, input_file, cutouts) #s right?

    # load previously trained Neural Network 
    model_found = False 
    for file in os.listdir(model_dir_name):
        # TEMP 
        #print(file)
        if file.startswith('model_60'): 
            #print(file)
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

    xs_best = []
    ys_best = []
    cn_prob = []

    
    start = time.time()
    cutouts_cleaned = []
    for cutout in cutouts: 
        inf_or_nan = np.isfinite(cutout)
        if False in inf_or_nan:
            pass
        elif cutout.min() < -200 or cutout.max() > 65536/3:
            pass
        else:
            cutouts_cleaned.append(cutout)

    cutouts_cleaned = np.array(cutouts_cleaned)
    # use std and mean to regularize cutout
    cutouts = regularize(cutouts_cleaned, mean, std)
    start_ml = time.time()
    output = model.predict(cutouts)
    end_ml = time.time()
    print('ML predictions completed in', round(end_ml-start_ml, 10), ' seconds')

    for i in range(len(cutouts)):
        good_probability = output[i][1]
        cn_prob.append(good_probability) 

    cn_prob, xs, ys, cutouts = zip(*sorted(zip(cn_prob, xs, ys, cutouts), reverse = True))

    plotted_stars = 0
    for i in range(len(cutouts)): 
        if plotted_stars < 25:
            good_probability = cn_prob[i]
            if good_probability > conf_cutoff:       
                xs_best.append(xs[i])
                ys_best.append(ys[i])
                plotted_stars += 1

    end = time.time()
    print('ML process completed in', round(end-start, 10), ' seconds')
            
    print(plotted_stars)

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