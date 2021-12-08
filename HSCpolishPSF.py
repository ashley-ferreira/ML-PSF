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


dir = '20191120'
inputFile = 'rS1i04545.fits'
if len(sys.argv)>2:
    dir = sys.argv[1]
    inputFile = sys.argv[2]



## load the fits saves
outFile = dir+'/'+inputFile.replace('.fits', '_savedFits.pickle')
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
