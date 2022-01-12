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


def HSCpolishPSF_main(dir='20191120', inputFile='rS1i04545.fits'):
 
    if len(sys.argv)>2:
        dir = sys.argv[1]
        inputFile = sys.argv[2]



    ## load the fits saves
    outFile = dir+'/'+inputFile.replace('.fits', '_cutouts_savedFits.pickle')
    with open(outFile, 'rb') as han:
        [stds, seconds, peaks, xs, ys, cutouts] = pick.load(han)

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

    top_5 = args[:5]
    top_10 = args[:10]
    top_15 = args[:15]
    top_20 = args[:20]
    top_30 = args[:30]
    top_35 = args[:35]


    #### generate the new psf

    with fits.open(dir+'/'+inputFile) as han:
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
    print(fwhm)

    newPSFFile = dir+'/psfStars/'+inputFile.replace('.fits','.goodPSF.fits')
    print('Saving to', newPSFFile)
    goodPSF.psfStore(newPSFFile, psfV2=True)

   
    cutoutWidth = max(30, int(5*fwhm))
    count = 0

    for x,y,std,second,d in zip(xs,ys,stds,seconds,dist):
        label = 0
        top_5_label = 0
        top_10_label = 0
        top_15_label = 0
        top_20_label = 0
        top_30_label = 0
        top_35_label = 0

        count +=1

        y_int = int(y)
        x_int = int(x)
        cutout = img_data[y_int-cutoutWidth:y_int+cutoutWidth+1, x_int-cutoutWidth:x_int+cutoutWidth+1]

        print(cutout.shape)

        cutouts.append(cutout)
        cutout_astropy = Cutout2D(img_data, (x,y), 2*cutoutWidth)

        if x in xs[top_5]:
            label = 1
            top_5_label = 1
            top_10_label = 1
            top_15_label = 1
            top_20_label = 1
            top_30_label = 1
            top_35_label = 1

        elif x in xs[top_10]:
            label=1
            top_10_label = 1
            top_15_label = 1
            top_20_label = 1
            top_30_label = 1
            top_35_label = 1
        
        elif x in xs[top_15]:
            label=1
            top_15_label = 1
            top_20_label = 1
            top_30_label = 1
            top_35_label = 1
            
        elif x in xs[top_20]:
            label = 1
            top_20_label = 1
            top_30_label = 1
            top_35_label = 1

        elif x in xs[best]:
            label = 1
            top_30_label = 1
            top_35_label = 1
        
        elif x in xs[top_30]:
            top_30_label = 1
            top_35_label = 1

        elif x in xs[top_35]:
            top_35_label = 1
        
        #else: 
        #    label = 0


        outFile = dir+'/NN_data_n=5/'+inputFile.replace('.fits', str(count) + '_cutoutData.pickle')
        print("Saving to", outFile)
        with open(outFile, 'wb+') as han:
            pick.dump([count, cutout, top_5_label], han)

        outFile = dir+'/NN_data_n=10/'+inputFile.replace('.fits', str(count) + '_cutoutData.pickle')
        print("Saving to", outFile)
        with open(outFile, 'wb+') as han:
            pick.dump([count, cutout, top_10_label], han)

        outFile = dir+'/NN_data_n=15/'+inputFile.replace('.fits', str(count) + '_cutoutData.pickle')
        print("Saving to", outFile)
        with open(outFile, 'wb+') as han:
            pick.dump([count, cutout, top_15_label], han)

        outFile = dir+'/NN_data_n=20/'+inputFile.replace('.fits', str(count) + '_cutoutData.pickle')
        print("Saving to", outFile)
        with open(outFile, 'wb+') as han:
            pick.dump([count, cutout, top_20_label], han)

        outFile = dir+'/NN_data_n=25/'+inputFile.replace('.fits', str(count) + '_cutoutData.pickle')
        print("Saving to", outFile)
        with open(outFile, 'wb+') as han:
            pick.dump([count, cutout, label], han)

        outFile = dir+'/NN_data_n=30/'+inputFile.replace('.fits', str(count) + '_cutoutData.pickle')
        print("Saving to", outFile)
        with open(outFile, 'wb+') as han:
            pick.dump([count, cutout, top_30_label], han)

        outFile = dir+'/NN_data_n=35/'+inputFile.replace('.fits', str(count) + '_cutoutData.pickle')
        print("Saving to", outFile)
        with open(outFile, 'wb+') as han:
            pick.dump([count, cutout, top_35_label], han)

    return 1