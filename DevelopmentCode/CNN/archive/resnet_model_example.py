from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn
def residual_block(x: Tensor, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= 1,
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)
    out = Add()([x, y])
    out = relu_bn(out)
    return out
def resnet(input_shape, unique_labs=2, num_filters=32, num_densefilters=32):
    inputs = Input(shape=input_shape)
    t = BatchNormalization()(inputs)
    num_blocks_list = [2, 2, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, filters=num_filters)
        num_filters *= 2
        t = MaxPool2D(pool_size=(2, 2), padding='valid')(t)
    t = Flatten()(t)
    outputs = Dense(num_densefilters, activation='softmax')(t)
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model