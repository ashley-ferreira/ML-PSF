import numpy as np, astropy.io.fits as pyf,pylab as pyl
from trippy import psf, pill, psfStarChooser
from trippy import scamp,MCMCfit
import scipy as sci
from os import path
import sys
from astropy.visualization import interval, ZScaleInterval

goodPSF = sys.argv[1]

file_dir = '/arc/home/ashley/HSC_May25-lsst/rerun/processCcdOutputs/03071/HSC-R2/corr'
inputFile = 'CORR-0218194-' + str(goodPSF) + '.fits'

zscale = ZScaleInterval()
comparePSF = file_dir+'/psfStars/'+inputFile #.replace('.fits','.goodPSF.fits')
otherPSF = psf.modelPSF(restore=comparePSF)
(o1, o2) = zscale.get_limits(otherPSF.lookupTable)
normer = interval.ManualInterval(o1,o2)
pyl.imshow(normer(otherPSF.lookupTable))
pyl.title('PSF for ' + inputFile)
pyl.show()