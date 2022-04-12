# NN-PSF: Almost Done
### Part of NSERC-CREATE NTCO training program astronomy research project
Student: Ashley Ferreira, Supervisor: Dr. Wesley Fraser

#### Purpose:

Given cutout of each source in an image along with their respective x and y coordinates, this program calls on a pre-trained machine learnining model that will return a subset of cutouts of sources to use for point spread function (PSF) creation. It also returns the x and y coordinates of these sources that can be used to pass into the python module TRIPPy in order to create the desired PSF.

A report or poster presentaion elaborating on the process, data used for training (where it can be accessed), and the results such as how well the algorithm performs is planned to be included as part of this project in the future.

#### Workflow:

There are two main components: 
1. FinalProgram: the final trained machine learning model as well as a program 'NN_PSF_star_chooser.py' that interfaces with the model for easiest use. 
2. DevelopmentCode: all of the code which was used to develop the final trained model. This code may be useful for others attempting to do similar work in the future. I have included the specific workflow below: 

>>> DataPrep/DataPrepLoop.py to create training/testing/validation data from images
>>> CNN/Star_NN_dev.py organizes data into training/testing/validation sets, trains and tests the model
>>> CNN/Star_NN_valid.py analyzes the performance of the model using the validation dataset, randomly chosen data from the larger set of data but which the model has never trained or tested on 
>>> CreatePSF/compare_NN_goodPSF.py a program which plots the chosen sources from the model along with the PSF that is create by TRIPPy using these sources. This is compared to the 25 stars chosen by a non-machine learning technique and the PSF that is created from those. 

#### Acknowledgments:

This work would not have been possible without TRIPPy and SExtractor, cited below:

Fraser, W. et al., 2016, To appear in AJ. DOI at Zenodo: http://dx.doi.org/10.5281/zenodo.48694

Bertin, E. & Arnouts, S. 1996: [SExtractor: Software for source extraction](https://ui.adsabs.harvard.edu/abs/1996A%26AS..117..393B/abstract), Astronomy & Astrophysics Supplement 317, 393 [BibTeX entry].

This research also used the Canadian Advanced Network For Astronomy Research (CANFAR) operated in partnership by the  Canadian Astronomy Data Centre and The Digital Research Alliance of Canada with support from the National Research Council of Canada, the Canadian Space Agency, CANARIE, and the Canadian Foundation for Innovation.

#### Future work ideas:

- Take whole image as input, not premade cutouts of sources
- Downsampling/other techniques to allow data from telescopes with significantly different pixel scales to be used
- ...
