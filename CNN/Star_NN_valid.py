import os
import sys
import pickle
from os import path

import tensorflow as tf
from keras import models, utils
import matplotlib as mpl
import matplotlib.pyplot as pyl
import numpy as np
from astropy.visualization import ZScaleInterval, interval
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels
from astropy.visualization import interval, ZScaleInterval
zscale = ZScaleInterval()

from optparse import OptionParser
parser = OptionParser()

np.random.seed(432)

pwd = '/arc/projects/uvickbos/ML-PSF/'
parser.add_option('-p', '--pwd', dest='pwd', 
        default=pwd, type='str', 
        help=', default=%default.')

model_dir = pwd + 'Saved_Model/' 
parser.add_option('-m', '--model_dir_name', dest='model_name', \
        default='default_model', type='str', \
        help='name for model directory, default=%default.')

cutout_size = 111
parser.add_option('-c', '--cutout_size', dest='cutout_size', \
        default=cutout_size, type='int', \
        help='c is size of cutout required, produces (c,c) shape, default=%default.')

parser.add_option('-t', '--training_subdir', dest='training_subdir', \
        default='NN_data_' + str(cutout_size) + '/', type='str', \
        help='subdir in pwd for training data, default=%default.')

def get_user_input():
    '''
    Gets user user preferences for neural network training parameters/options

    Parameters:    

        None

    Returns:

        model_dir_name (str): directory to where trained model is stored

        cutout_size (int): is size of cutout required, produces (cutout_size,cutout_size) shape

        pwd (str): working directory, will load data from subdir and save model into subdir

        training_sub_dir (str): subdir in pwd for training data

    '''
    (options, args) = parser.parse_args()

    model_dir_name = model_dir + options.model_name
    
    return model_dir_name, options.cutout_size, options.pwd, options.training_subdir


def load_presaved_data(cutout_size, model_dir_name):
    '''
    Create presaved data file to use for neural network training

    Parameters:    

        cutout_size (int): defines shape of cutouts (cutout_size, cutout_size)

        model_dir_name (str): directory to load data and save regularization params

    Returns:
        
        data (lst), which consists of:

            cutouts (arr): 3D array conisting of 2D image data for each cutout

            labels (arr): 1D array containing 0 or 1 label for bad or good star respectively

            xs (arr): 1D array containing central x position of cutout 

            ys (arr): 1D array containing central y position of cutout 

            fwhms (arr): 1D array containing fwhm values for each cutout 
            
            files (arr): 1D array containing file names for each cutout

    '''

    with open(model_dir_name + 'WITHHELD_' + str(cutout_size) + '_presaved_data.pickle', 'rb') as han:
        [cutouts, labels, xs, ys, fwhms, files] = pickle.load(han) 

    cutouts = np.asarray(cutouts).astype('float32')
    std = np.nanstd(cutouts)
    mean = np.nanmean(cutouts)
    cutouts -= mean
    cutouts /= std
    w_bad = np.where(np.isnan(cutouts))
    cutouts[w_bad] = 0.0

    with open(model_dir + 'regularization_data.pickle', 'rb') as han:
        [std, mean] = pickle.load(han)

    cutouts = np.asarray(cutouts).astype('float32')
    cutouts -= mean
    cutouts /= std
    w_bad = np.where(np.isnan(cutouts))
    cutouts[w_bad] = 0.0

    return [cutouts, labels, xs, ys, fwhms, files]


def validate_CNN(model_dir_name, data):
    '''
    Parameters:    

        model_dir_name (str): directory where trained model is saved

        data (lst), which consists of:

            cutouts (arr): 3D array conisting of 2D image data for each cutout

            labels (arr): 1D array containing 0 or 1 label for bad or good star respectively

            xs (arr): 1D array containing central x position of cutout 

            ys (arr): 1D array containing central y position of cutout 

            fwhms (arr): 1D array containing fwhm values for each cutout 
            
            files (arr): 1D array containing file names for each cutout

    Returns:
        
        None
    
    '''
    # section for setting up hyperparameters
    batch_size = 16 

    # unpack presaved data
    cutouts, labels, xs, ys, fwhms, files = data[0], data[1], data[2], data[3], data[4], data[5]

    # load model                         
    model_found = False 
    for file in os.listdir(model_dir_name):
        if file.startswith('model_'):
            cn_model = models.load_model(model_dir_name + file)
            model_found = True
            break
    if model_found == False: 
        print('ERROR: no model file in', model_dir_name)
        sys.exit()

    X_valid = cutouts
    y_valid = labels
    unique_labs = int(len(np.unique(y_valid)))
    y_valid_binary = utils.np_utils.to_categorical(y_valid, unique_labs)
    X_valid = np.asarray(X_valid).astype('float32')
    preds_valid = cn_model.predict(X_valid, verbose=1)

    test_good_p = []
    for p in preds_valid:
        test_good_p.append(p[1])
        
    bins = np.linspace(0, 1, 100)
    pyl.hist(test_good_p, label = 'validation set confidence', bins=bins, alpha=0.5, density=True)
    pyl.xlabel('Good Star Confidence')
    pyl.ylabel('Count')
    pyl.legend(loc='best')
    pyl.show()
    pyl.close()
    pyl.clf()

    results = cn_model.evaluate(X_valid, y_valid_binary, batch_size=batch_size)
    print("validation loss, validation acc:", results)

    zscale = ZScaleInterval()

    X_valid = np.squeeze(X_valid, axis=3)
    print(X_valid.shape) 

    # plot confusion matrix
    fig2 = pyl.figure()
    y_valid_binary = np.argmax(y_valid_binary, axis = 1)
    preds_valid_binary = np.argmax(preds_valid, axis = 1)
    cm = confusion_matrix(y_valid_binary, preds_valid_binary)
    pyl.matshow(cm, cmap=mpl.cm.cool)
    for (i, j), z in np.ndenumerate(cm):
        pyl.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    pyl.title('Confusion matrix (validation data)')
    pyl.colorbar(cmap=mpl.cm.cool)
    pyl.xlabel('Predicted labels')
    pyl.ylabel('True labels')
    pyl.show()
    pyl.clf()

    # plot of FWHMs
    fwhms_test_misclass = []
    for i in range(len(preds_valid)):
        if y_valid[i] == 1 and preds_valid[i][0] > 0.5:
            fwhms_test_misclass.append(fwhms[i])
            '''
            (c1, c2) = zscale.get_limits(y_valid[i])
            normer3 = interval.ManualInterval(c1,c2)
            pyl.title('labeled good star, predicted bad star at conf=' + str(preds_valid[i][1]))
            pyl.imshow(normer3(X_valid[i]))
            pyl.show()
            pyl.close()
            '''
        elif y_valid[i] == 0 and preds_valid[i][1] > 0.5:
            fwhms_test_misclass.append(fwhms[i])
            '''
            (c1, c2) = zscale.get_limits(X_valid[i])
            normer5 = interval.ManualInterval(c1,c2)
            pyl.title('labeled bad star, predicted good star at conf=' + str(preds_valid[i][1])) 
            pyl.imshow(normer5(X_valid[i]))
            pyl.show()
            pyl.close()
            '''
    pyl.hist(fwhms, label = 'FWHM of full validation set', bins='auto', alpha=0.5) 
    pyl.hist(fwhms_test_misclass, label = 'FWHM of misclassed validation set', bins='auto', alpha=0.5, color='pink') 
    pyl.xlabel('FWHM')
    pyl.ylabel('Count')
    pyl.legend(loc='best')
    pyl.show()
    pyl.close()
    pyl.clf()

    # accuracy vs confidence plot
    confidence_step = 0.001 # likely automatic way to do this but i didn't easily find
    confidence_queries = np.arange(confidence_step, 1, confidence_step) 
    good_star_acc = []
    bad_star_acc = []
    recall = []
    precision = []
    fp_rate = []

    for c in confidence_queries:
        good_stars_correct = 0
        good_stars_incorrect = 0
        good_stars_above_c = 0
        bad_stars_correct = 0
        bad_stars_incorrect = 0
        bad_stars_above_c = 0

        for i in range(len(preds_valid)):
            '''
            if y_test[i] == 0 and preds_test[i][1] > c:
                #(c1, c2) = zscale.get_limits(X_test[i])
                #normer5 = interval.ManualInterval(c1,c2)
                #pyl.title('labeled bad star, predicted good star at conf=' + str(preds_test[i][1])) # so great you already have this
                #pyl.imshow(normer5(X_test[i]))
                #pyl.show()
                #pyl.close()
                misclass_80p += 1
            elif y_test[i] == 1 and preds_test[i][1] > c:
                good_class_80p += 1
            '''
            '''
            if preds_test[i][0] > c:
                bad_stars_above_c +=1
                if y_test[i] == 0:
                    bad_stars_correct +=1
                elif y_test[i] == 1:
                    bad_stars_incorrect +=1
            elif preds_test[i][1] > c:
                good_stars_above_c +=1 # wrong way? out of all preds?
                if y_test[i] == 1:
                    good_stars_correct +=1 
                elif y_test[i] == 0:
                    good_stars_incorrect +=1      
            '''
            if preds_valid[i][1] > c:
                good_stars_above_c +=1 
                if y_valid[i] == 1:
                    good_stars_correct +=1 
                elif y_valid[i] == 0:
                    good_stars_incorrect +=1
            else:
                bad_stars_above_c +=1
                if y_valid[i] == 0:
                    bad_stars_correct +=1
                elif y_valid[i] == 1:
                    bad_stars_incorrect +=1
                    

        print('good', good_stars_correct, good_stars_incorrect, good_stars_above_c)
        print('bad', bad_stars_correct, bad_stars_incorrect, bad_stars_above_c)
        good_star_acc.append(good_stars_correct/good_stars_above_c)
        bad_star_acc.append(bad_stars_correct/bad_stars_above_c)
        # double check recall and precision calculations, switch to fp..
        recall.append(good_stars_correct/(good_stars_correct+bad_stars_incorrect)) 
        fp_rate.append(bad_stars_incorrect/(bad_stars_incorrect+bad_stars_correct)) 
        precision.append(good_stars_correct/(good_stars_correct+good_stars_incorrect))

    pyl.title('Accuracy Curve')
    pyl.plot(confidence_queries, good_star_acc, label='good star classificantion')
    pyl.plot(confidence_queries, bad_star_acc, label='bad star clasification')
    pyl.legend()
    pyl.xlabel('Confidence cutoff for good star classification')
    pyl.ylabel('Accuracy')
    pyl.show()
    pyl.close()
    pyl.clf()

    xy = np.arange(0,1, confidence_step)
    #perfect_ROC = np.concatenate(([0],np.ones(int(1/confidence_step)-1)))
    perfect_ROC = np.ones(len(xy))
    perfect_ROC[0] = 0

    pyl.title('ROC Curve')
    pyl.plot(xy, xy, '--', label='random chance refence line')
    pyl.plot(xy, perfect_ROC, '--', label='perfect classifier')
    pyl.plot(fp_rate, recall, label='trained CNN') # fp too big
    pyl.legend()
    pyl.xlabel('False Positive Rate')
    pyl.ylabel('True Positive Rate (Recall)')
    pyl.show()
    pyl.close()
    pyl.clf()

    #perfect_PR = np.concatenate((np.ones(len(confidence_queries)-1), [0]))
    perfect_PR = np.ones(len(xy))
    perfect_PR[len(xy)-1] = 0

    pyl.title('PR Curve')
    pyl.plot(xy, perfect_PR, '--', label='perfect classifier')
    pyl.plot(recall, precision, label='trained CNN')
    pyl.legend()
    pyl.xlabel('Recall')
    pyl.ylabel('Precision')
    pyl.show()
    pyl.close()
    pyl.clf()

def main():
    model_dir_name, cutout_size, pwd, training_subdir = get_user_input()

    if True:#try:
        data = load_presaved_data(cutout_size, model_dir_name)
        validate_CNN(model_dir_name, data)

    '''
    except Exception as Argument:
        print('Star_NN_valid.py' + str(Argument))

        # creating/opening a file
        err_log = open(model_dir_name + 'error_log.txt', 'a')

        # writing in the file
        err_log.write('Star_NN_valid.py' + str(Argument))
        
        # closing the file
        err_log.close()  
    '''

if __name__ == '__main__':
    main()