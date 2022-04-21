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

# loop through each source and create new cutout files that include
# that are labelled 1 if in top 25 determined by goodPSF or
# labelled 0 otherwise, this data is exactly what the CNN uses
for x,y,cutout in zip(xs,ys,cutouts): 

    if x in xs[best]:
        label = 1
    else: 
        label = 0