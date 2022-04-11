from HSCgetStars_func import HSCgetStars_main 

# give it a filename 
# will make cutouts if needed?
# will pick best cutouts with function 
# send to trippy?

# no even more basic, just function that already has
# model saved and then calls on trippy for any given 
# set of coutotuts from image??


#def load_data():



#def load_model():



#def use_trippy():


pwd = '/arc/projects/uvickbos/ML-PSF/'
parser.add_option('-p', '--pwd', dest='pwd', 
        default=pwd, type='str', 
        help=', default=%default.')

model_dir = pwd + 'Saved_Model/' 
parser.add_option('-m', '--model_dir_name', dest='model_name', \
        default='default_model/', type='str', \
        help='name for model directory, default=%default.')

parser.add_option('-c', '--conf_cutoff', dest='conf_cutoff', 
        default='0.95', type='float', \
        help='confidence cutoff, default=%default.')

parser.add_option('-S', '--SNR_proxy_cutoff', dest='SNR_proxy_cutoff', 
        default='10.0', type='float', 
        help='SNR proxy cutoff, default=%default.')

parser.add_option('-s', '--min_num_stars', dest='min_num_stars', 
        default='10', type='int', 
        help='minimum number of stars acceptable, default=%default.')


def regularize(cutout, mean, std):
    '''
    Regularizes either single cutout or array of cutouts

    Parameters:

        mean (float): mean used in training data regularization  

        std (float): std used in training data regularization

    Returns:

        regularized_cutout (arr): regularized cutout
    
    '''
    cutout -= mean
    cutout /= std
    w_bad = np.where(np.isnan(cutout))
    cutout[w_bad] = 0.0
    regularized_cutout = cutout

    return regularized_cutout


def NN_PSF_star_chooser(cutouts, min_num_stars, SNR_proxy_cutoff, conf_cutoff):
    '''
    Final product of NN-PSF project

    Requirements: provided on backend from trippy

        folder NN_PSF which contains:
            NN_PSF_model: 
            regularization_data:


    Parameters: provided by user   

        cutouts (arr): 3D array conisting of 2D image data for each cutout
                        FOR THIS MODEL MUST BE (111,111) SIZE

    Returns:
        
        None
    
    '''
    # crop to right size?




# read in cutout data for input_file
    outFile = file_dir+input_file.replace('.fits', str(cutout_size) + \
         '_metadata_cutouts_savedFits.pickle')

    with open(outFile, 'rb') as han:
        [std, seconds, peaks, xs, ys, cutouts, fwhm, inputFile] = pickle.load(han)

    # load previously trained Neural Network 
    model_found = False 
    for file in os.listdir(model_dir_name):
        if file.startswith('model_'):
            model = keras.models.load_model(model_dir_name + file)
            model_found = True
            break
    if model_found == False: 
        print('ERROR: no model file in', model_dir_name)
        sys.exit()

    # load training set std and mean
    with open(model_dir + 'regularization_data.pickle', 'rb') as han:
        [std, mean] = pickle.load(han)

    # use std and mean to regularize cutout
    cutouts = regularize(cutouts, mean, std)