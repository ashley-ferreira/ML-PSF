import pickle as pick, numpy as np, pylab as pyl
from sklearn import cluster
from astropy.visualization import interval, ZScaleInterval
from trippy import psf, psfStarChooser
from astropy.io import fits
import sys

## not used for now, but kept for possible use later
def getCluster(stds, seconds, eps=0.01, min_samples=5 ):
    x = stds/np.std(stds)
    y = seconds/np.std(seconds)
    print(len(x))

    X = np.zeros((len(x), 2))
    X[:, 0] = x
    X[:, 1] = y
    clust = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clust.labels_

    u_labels = np.unique(labels)
    print(labels)
    print(u_labels)
    clump_stds = []
    for i, label in enumerate(u_labels[1:]):
        w = np.where(labels == label)
        clump_stds.append(np.mean(stds[w]))
    print(clump_stds/np.std(stds))
    return (clust, labels, clump_stds)


def HSCpolishPSF_main(fixed_cutout_len = 0, dir='20191120', inputFile='rS1i04545.fits', cutout_file=''):
 
    #if len(sys.argv)>2:
    #    dir = sys.argv[1]
    #    inputFile = sys.argv[2]



    ## load the fits saves
    #if fixed_cutout_len != 0:
    #   outFile = dir+'/'+inputFile.replace('.fits', str(fixed_cutout_len) + '_cutouts_savedFits.pickle')
    #else:
    try:
            
        outFile = dir+'/'+inputFile.replace('.fits', '111_metadata_cutouts_savedFits.pickle')
        with open(outFile, 'rb') as han:
            [stds, seconds, peaks, xs, ys, cutouts, fwhm, inputFile] = pick.load(han)

        metadata_dict = {}
        metadata_dict['stds'] = stds 
        metadata_dict['seconds'] = seconds 
        metadata_dict['peaks'] = peaks 
        metadata_dict['xs'] = xs 
        metadata_dict['ys'] = ys 
        metadata_dict['fwhm'] = fwhm 
        metadata_dict['inputFile'] = inputFile

        
        len_c = len(cutouts)
        print(cutouts.shape)
        if cutouts.shape == (len_c, 111,111): # some non 111,111 let in from getstars
            zscale = ZScaleInterval()

            ## select only those stars with really low STD
            w = np.where(stds/np.std(stds)<0.001)
            stds = stds[w]
            seconds = seconds[w]
            peaks = peaks[w]
            xs = xs[w]
            ys = ys[w]
            cutouts = np.array(cutouts)[w]

            s = np.std(stds)

            ##find the best 25 stars (the closest to the origin in weighted STD and second highest pixel value)
            dist = ((stds/s)**2 + (seconds/peaks)**2)**0.5
            args = np.argsort(dist)
            best = args[:25]

            #### generate the new psf
            with fits.open(dir+'/'+inputFile) as han:
                img_data = han[1].data
                header = han[0].header

            starChooser=psfStarChooser.starChooser(img_data,
                                                xs[best],ys[best],
                                                xs[best]*500,xs[best]*1.0)

            try:
                (goodFits, goodMeds, goodSTDs) = starChooser(30,200,noVisualSelection=True,autoTrim=False,
                                                            bgRadius=15, quickFit = False,
                                                            printStarInfo = True,
                                                            repFact = 5, ftol=1.49012e-08)
            except:
                print('error with starChooser, conitnuing to next cutout')
                return 0

            goodPSF = psf.modelPSF(np.arange(61),np.arange(61), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
            goodPSF.genLookupTable(img_data, goodFits[:,4], goodFits[:,5], verbose=False)
            fwhm = goodPSF.FWHM()
            print(fwhm)

            newPSFFile = dir+'/psfStars/'+inputFile.replace('.fits','.metadata_goodPSF.fits')
            print('Saving to', newPSFFile)
            goodPSF.psfStore(newPSFFile, psfV2=True)

            cutoutWidth = fixed_cutout_len // 2 # or adapt to f*fwhm if zero
            count = 0
            for x,y in zip(xs,ys): 
                y_int = int(y)
                x_int = int(x)
                cutout = img_data[y_int-cutoutWidth:y_int+cutoutWidth+1, x_int-cutoutWidth:x_int+cutoutWidth+1]

                print(cutout.shape)

                label = 0
                count +=1 
                if x in xs[best]: # just redo the cutout
                    label = 1

                final_file = dir+'/NN_data_metadata_' + str(fixed_cutout_len) + '/'+inputFile.replace('.fits', str(count) + '_metadata_cutoutData.pickle')
                with open(final_file, 'wb+') as han:
                    pick.dump([count, cutout, label, metadata_dict], han)

        return 1

    except Exception as e: 
        print('FAILURE')
        print(e)

        return 0