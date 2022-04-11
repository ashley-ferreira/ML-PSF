import pickle as pick, numpy as np, pylab as pyl
from sklearn import cluster
from astropy.visualization import interval, ZScaleInterval
from trippy import psf, psfStarChooser
from astropy.io import fits
import sys

def HSCpolishPSF_main(file_dir, input_file, cutout_file, fixed_cutout_len, training_dir):
    '''
    Given cutouts of all the sources in an image, this function selects the best
    25 stars in an image from weighted STD and second highest pixel value. These 
    cutouts are then resaved with those that fall within the top 25 labelled as 
    such to be used in neural network training. From these 25 stars the function 
    also generates and saves new PSF called goodPSF.
    
    Adapted from:  https://github.com/fraserw

    Parameters:    

        file_dir (str): directory where inputFile is saved

        input_file (str): original image file

        cutout_file (str): cutouts of all sources in image created from 
                           HSC_getStars_main function

        fixed_cutout_len (int): force cutouts to have shape
                                (fixed_cutout_len, fixed_cutout_len)

    Returns:
        
        None

    '''
    #   outFile = dir+'/'+inputFile.replace('.fits', str(fixed_cutout_len) + '_cutouts_savedFits.pickle')
    try:
        # read in saved cutout file created from HSCgetStars_main    
        with open(cutout_file, 'rb') as han:
            [stds, seconds, peaks, xs, ys, cutouts, fwhm, inputFile] = pick.load(han)

        # create dictionairy to store metadata
        metadata_dict = {}
        metadata_dict['stds'] = stds 
        metadata_dict['seconds'] = seconds 
        metadata_dict['peaks'] = peaks 
        metadata_dict['xs'] = xs 
        metadata_dict['ys'] = ys 
        metadata_dict['fwhm'] = fwhm 
        metadata_dict['inputFile'] = inputFile
        
        # make sure cutouts are all of correct shape
        if cutouts.shape == (len(cutouts), fixed_cutout_len, fixed_cutout_len): 

            ## select only those stars with really low STD
            w = np.where(stds/np.std(stds)<0.001)
            stds = stds[w]
            seconds = seconds[w]
            peaks = peaks[w]
            xs = xs[w]
            ys = ys[w]
            cutouts = np.array(cutouts)[w]
            s = np.std(stds)

            ## find the best 25 stars (the closest to the origin in 
            ## weighted STD and second highest pixel value)
            dist = ((stds/s)**2 + (seconds/peaks)**2)**0.5
            args = np.argsort(dist)
            best = args[:25]

            ## generate the new psf called goodPSF
            with fits.open(file_dir+'/'+input_file) as han:
                img_data = han[1].data
                header = han[0].header

            starChooser=psfStarChooser.starChooser(img_data,
                                                xs[best],ys[best],
                                                xs[best]*500,xs[best]*1.0)

            (goodFits, goodMeds, goodSTDs) = starChooser(30,200,noVisualSelection=True,autoTrim=False,
                                                        bgRadius=15, quickFit = False,
                                                        printStarInfo = True,
                                                        repFact = 5, ftol=1.49012e-08)

            goodPSF = psf.modelPSF(np.arange(61),np.arange(61), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
            goodPSF.genLookupTable(img_data, goodFits[:,4], goodFits[:,5], verbose=False)
            fwhm = goodPSF.FWHM()

            newPSFFile = file_dir+'/psfStars/'+input_file.replace('.fits','._goodPSF.fits')
            print('Saving to', newPSFFile)
            goodPSF.psfStore(newPSFFile, psfV2=True)

            # loop through each source and create new cutout files that include
            # that are labelled 1 if in top 25 determined by goodPSF or
            # labelled 0 otherwise, this data is exactly what the CNN uses
            count = 0
            for x,y,cutout in zip(xs,ys,cutouts): 
                count +=1 

                if x in xs[best]:
                    label = 1
                else: 
                    label = 0

                final_file = training_dir \
                             + '/' + input_file.replace('.fits', '_cutout_' + str(count) \
                             + '_cutoutData.pickle')

                with open(final_file, 'wb+') as han:
                    pick.dump([count, cutout, label, y, x, fwhm, inputFile], han)
        else:
            print('Cutouts of wrong shape:', cutouts.shape)
    
    except Exception as Argument:
        print('HSCpolishPSF.py' + Argument)

        # creating/opening a file
        err_log = open(training_dir + 'data_prep_error_log.txt', 'a')

        # writing in the file
        err_log.write('HSCpolishPSF.py' + input_file + str(Argument))
        
        # closing the file
        err_log.close()  