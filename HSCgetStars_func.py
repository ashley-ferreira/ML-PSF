from trippy import psf, scamp, psfStarChooser, bgFinder, MCMCfit
from astropy.io import fits
from astropy.visualization import interval, ZScaleInterval
import numpy as np, pylab as pyl
import pickle as pick
import sys, os


def HSCgetStars_main(dir = '20191120', inputFile = 'rS1i04545.fits', psfFile = 'rS1i04545.psf.fits'):

    print(len(sys.argv))
    if len(sys.argv)>3:
        dir = sys.argv[1]
        inputFile = sys.argv[2]
        psfFile = sys.argv[3]

    ### open the fits image
    with fits.open(dir+'/'+inputFile) as han:
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
    scamp.makeParFiles.writeParam(numAps=1) #numAps is thenumber of apertures that you want to use. Here we use 1

    fits.writeto('junk.fits', img_data, header=header, overwrite=True)
    scamp.runSex('HSC.sex', 'junk.fits' ,options={'CATALOG_NAME':f'{dir}/{inputFile}.cat'},verbose=True)
    catalog = scamp.getCatalog(f'{dir}/{inputFile}.cat',paramFile='def.param')
    os.system('rm junk.fits')


    ## select only high SNR stars
    w = np.where(catalog['FLUX_AUTO']/catalog['FLUXERR_AUTO'] > 200)
    X_ALL = catalog['XWIN_IMAGE'][w]
    Y_ALL = catalog['YWIN_IMAGE'][w]


    ## load the PSF that Wes generated at an earlier point. This is not a great PSF!
    goodPSF = psf.modelPSF(restore=dir+'/'+psfFile)
    fwhm = goodPSF.FWHM()
    print(fwhm)

    cutoutWidth = max(30, int(5*fwhm))

    zscale = ZScaleInterval()


    ## fit each star with the PSF to get its brightness and position.
    stds, seconds, peaks, xs, ys = [], [], [], [], []
    cutouts = []
    rem_cutouts = []
    for i in range(len(X_ALL)):
        #if len(xs)>25: break


        x, y = X_ALL[i], Y_ALL[i]
        print(f'Fitting good source {x}, {y}')

        x_int, y_int = int(x), int(y)
        if not (x_int>cutoutWidth+2 and y_int>cutoutWidth+2 and x_int<(B-cutoutWidth-2) and y_int<(A-cutoutWidth-2)):
            continue

        cutout = img_data[y_int-cutoutWidth:y_int+cutoutWidth+1, x_int-cutoutWidth:x_int+cutoutWidth+1]

        peak = np.max(cutout[cutoutWidth-1:cutoutWidth+2, cutoutWidth-1:cutoutWidth+2])
        peaks.append(peak)

        #background estimate
        bgf = bgFinder.bgFinder(cutout)
        bg_estimate = bgf()


        try:
            fitter = MCMCfit.LSfitter(goodPSF, cutout)
            fitPars = fitter.fitWithModelPSF(cutoutWidth, cutoutWidth, m_in=10.0,
                                            fitWidth = 7, ftol = 1.49e-6,
                                            verbose = True)
            if fitPars[2]<=0:
                fitPars=None
        except:
            fitPars = None

        if fitPars  is not None:

            print(fitPars)
            (aa,bb) = cutout.shape

            model_cutout = goodPSF.plant(fitPars[0], fitPars[1], fitPars[2], cutout*0.0,returnModel=True,addNoise=False)
            pixel_weights = 1.0/(np.abs(model_cutout)+1.0)
            pixel_weights /= np.sum(pixel_weights)

            rem_cutout = goodPSF.remove(fitPars[0], fitPars[1], fitPars[2], cutout,useLinePSF=False)

            weighted_std = np.sum(pixel_weights*(rem_cutout - np.mean(rem_cutout))**2)/np.sum(model_cutout)*(aa*bb)

            sorted = np.sort(rem_cutout.reshape(aa*bb))
            second_highest = sorted[-2]


            stds.append(weighted_std)
            seconds.append(second_highest)
            xs.append(x)
            ys.append(y)
            cutouts.append(cutout[:])
            rem_cutouts.append(rem_cutout[:])
            #if x in goodFits[:, 4]:
            #    w = np.where(goodFits[:,4]==x)
            #    print(w)
            #    Fits.append([goodFits[:, 2][w], goodFits[:, 3][w]])
            #else:
            #    Fits.append([-1, -1])

            print('Second Highest', second_highest, stds[-1])
            print()

    std = np.array(stds)
    seconds = np.array(seconds)
    xs = np.array(xs)
    ys = np.array(ys)
    peaks = np.array(peaks)
    cutouts = np.array(cutouts)
    #Fits = np.array(Fits)


    ## save the fits data to file.
    # save original cutouts
    outFile = dir+'/'+inputFile.replace('.fits', '_cutouts_savedFits.pickle')
    print("Saving to", outFile)
    with open(outFile, 'wb+') as han:
        pick.dump([std, seconds, peaks, xs, ys, cutouts], han)
    # save cutouts with PSF removed
    outFile = dir+'/'+inputFile.replace('.fits', 'rem_cutouts_savedFits.pickle')
    print("Saving to", outFile)
    with open(outFile, 'wb+') as han:
        pick.dump([std, seconds, peaks, xs, ys, cutouts], han)



    #if len(sys.argv)>2:
    #    exit()

    fig1 = pyl.figure(1)
    pyl.scatter(stds, seconds)
    pyl.xlabel('Stds')
    pyl.ylabel('Second Peaks')

    fig2 = pyl.figure(2)
    pyl.scatter(stds/np.std(stds), seconds/np.std(seconds))
    pyl.xlabel('Stds')
    pyl.ylabel('Second Peaks')

    pyl.show()

    return 1