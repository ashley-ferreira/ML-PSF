import pickle as pick, numpy as np, pylab as pyl
from sklearn import cluster
from astropy.visualization import interval, ZScaleInterval
from trippy import psf, psfStarChooser
from astropy.io import fits
import sys
import time, os
from trippy import psf, scamp, psfStarChooser, bgFinder, MCMCfit

#def HSCpolishPSF_main(file_dir, input_file, cutout_file, fixed_cutout_len, training_dir):
def non_ML_timing(file_dir, input_file, cutout):
    '''
    This function uses Source-Extractor to create cutouts from all the sources 
    in an image.
    
    Adapted from:  https://github.com/fraserw

    Parameters:    

        file_dir (str): directory where inputFile is saved

        input_file (str): original image file

        cutout_file (str): filename to save cutouts of all sources in image 

        fixed_cutout_len (int): force cutouts to have shape
                                (fixed_cutout_len, fixed_cutout_len)

                        --> set to zero for cutoutWidth = max(30, int(5*fwhm))

    Returns:
        
        None

    '''

    ### open the fits image
    with fits.open(file_dir+'/'+input_file) as han:
        img_data = han[1].data.astype('float64')
        header = han[0].header
    (A,B)  = img_data.shape


    catalog = scamp.getCatalog(f'{file_dir}/{input_file}.cat',paramFile='def.param')
    os.system('rm junk.fits')

    ## select only high SNR stars
    w = np.where(catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO'] > 200)
    X_ALL = catalog['XWIN_IMAGE'][w]
    Y_ALL = catalog['YWIN_IMAGE'][w]

    ## load the PSF that Wes generated at an earlier point. This is not a great PSF!
    file_psf = input_file.replace('.fits','.psf_cleaned.fits')
    goodPSF = psf.modelPSF(restore=file_dir+'psfStars/'+file_psf)

    cutoutWidth = 111//2

    ## fit each star with the PSF to get its brightness and position.
    start = time.time()
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
    
    start = time.time()
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

    # end after label part (what do you need from beginning?)
    end = time.time()
    print('non-ML process completed in', round(end-start, 10), ' seconds')