from trippy import psf, scamp, psfStarChooser, bgFinder, MCMCfit
from astropy.io import fits
from astropy.visualization import interval, ZScaleInterval
import numpy as np, pylab as pyl
import pickle as pick
import sys, os


def HSCgetStars_main(file_dir, input_file, cutout_file, fixed_cutout_len = 111):
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
    catalog = scamp.getCatalog(f'{file_dir}/{input_file}.cat',paramFile='def.param')
    os.system('rm junk.fits')


    ## select only high SNR stars
    w = np.where(catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO'] > 200)
    X_ALL = catalog['XWIN_IMAGE'][w]
    Y_ALL = catalog['YWIN_IMAGE'][w]


    ## load the PSF that Wes generated at an earlier point. This is not a great PSF!
    file_psf = input_file.replace('.fits','.psf_cleaned.fits')
    goodPSF = psf.modelPSF(restore=file_dir+file_psf)
    fwhm = goodPSF.FWHM()
    print('#################### FWHM ######################')
    print(fwhm)

    if fixed_cutout_len == 0:
        cutoutWidth = max(30, int(5*fwhm))
    else:
        cutoutWidth = fixed_cutout_len//2


    ## fit each star with the PSF to get its brightness and position.
    stds, seconds, peaks, xs, ys = [], [], [], [], []
    cutouts = []
    rem_cutouts = []
    for i in range(len(X_ALL)):
        x, y = X_ALL[i], Y_ALL[i]
        
        x_int, y_int = int(x), int(y)
        if not (x_int>cutoutWidth+2 and y_int>cutoutWidth+2 and \
                x_int<(B-cutoutWidth-2) and y_int<(A-cutoutWidth-2)):
            continue

        cutout = img_data[y_int-cutoutWidth:y_int+cutoutWidth+1,x_int-cutoutWidth:x_int+cutoutWidth+1]
        
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
                cutouts.append(cutout[:])
                rem_cutouts.append(rem_cutout[:])
            
            else:
                print('Cutout of wrong shape:', cutout.shape)

    # store everything in numpy array format
    std = np.array(stds)
    seconds = np.array(seconds)
    xs = np.array(xs)
    ys = np.array(ys)
    peaks = np.array(peaks)
    cutouts = np.array(cutouts)

    ## save the fits data to file.
    # save original cutouts
    outFile = file_dir+input_file.replace('.fits', '_' + str(fixed_cutout_len) + 
                                                    '_cutouts_savedFits.pickle')
    print("Saving to", outFile)
    with open(outFile, 'wb+') as han:
        pick.dump([std, seconds, peaks, xs, ys, cutouts, fwhm, input_file], han)
    # save cutouts with PSF removed
    outFile = dir+'/'+input_file.replace('.fits', '_' + str(fixed_cutout_len) +  
                                                '_rem_cutouts_savedFits.pickle')
    print("Saving to", outFile)
    with open(outFile, 'wb+') as han:
        pick.dump([std, seconds, peaks, xs, ys, cutouts, fwhm, input_file], han)