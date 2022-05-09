from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.layers.core import Dropout

def convnet_model_lesslayers(input_shape, unique_labs=2, dropout_rate=0.2):
    '''
    Defines the 2D Convolutional Neural Network (CNN)

    Parameters:    

        input_shape (arr): input shape for network

        training_labels (arr): training labels

        unique_labels (int): number unique labels (for good and bad stars = 2)

        dropout_rate (float): dropout rate

    Returns:
        
        model (keras model class): convolutional neural network to train

    '''

    model = Sequential()

    #hidden layer 1
    model.add(Conv2D(filters=8, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    #hidden layer 2 with Pooling
    #model.add(Conv2D(filters=8, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    #model.add(Dropout(dropout_rate))
    #model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    model.add(BatchNormalization())

    #hidden layer 3 with Pooling
    model.add(Conv2D(filters=8, input_shape=input_shape, activation='relu', padding='same', kernel_size=(3,3)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(unique_labs, activation='softmax')) 
    #model.add(Activation("softmax"))

    return model