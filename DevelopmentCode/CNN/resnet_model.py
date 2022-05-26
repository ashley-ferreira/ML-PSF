from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, Add, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPool2D, Concatenate
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

#class ResNet(nn.Module):
#    def __init__(self, block, layers, image_channels, num_classes):

def identity_block(self, x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=-1)(x)
    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x
def convolutional_block(self, x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=-1)(x)
    # Processing Residue with conv(1,1)
    x_skip = Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

    #how to add thest to self?

def convnet_model_resnet(input_shape, num_dense_nodes = 2, unique_labels=2, dropout_rate=0.2):

    # init function
    #self.input_shape = input_shape
    #self.identity_block(x, filter_size)
    #self.convolutional_block(x, filter_size)

    model = Sequential()

    # Step 2 (Initial Conv layer along with maxPool)
    x = Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [2, 2]#[3, 4, 6, 3]
    filter_size = 16
    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    #x = AveragePooling2D((2,2), padding = 'same')(x)
    x = MaxPool2D(pool_size=(2,2), padding = 'same')(x)
    x = Flatten()(x)
    x = Dense(num_dense_nodes, activation = self.activation)(x)
    #x = Dense(self.num_dense_nodes, activation = self.activation)(x)
    output = Dense(unique_labels, activation = 'softmax')(x)

    return output 