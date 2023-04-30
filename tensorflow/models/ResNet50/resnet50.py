"""
Implementing ResNet50 from scratch.
Author: Mrinal Kanti Dhar
"""

from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization  
from tensorflow.keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model

def conv_block(input, filters:int, kernel_size:tuple=(1,1), strides:tuple=(1,1), 
               padding:str='valid', bn_axis:int=3, activation:str=None):
    
    """ Performs either convolution-batch normalization-relu or convolution-batch normalization.    
    
    Inputs
    -------------
    input: An input tensor
    filters (int): No. of filters
    kernel_size (tuple): Size of each kernel
    strides (tuple): Strides for tranlating the kernel
    padding(str): valid or same
    bn_axis (int): Batch normalization axis
    activation (str): Activation function
    
    Outputs
    --------------
    x: An output tensor
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    x = BatchNormalization(axis=bn_axis)(x)
    
    if activation is not None:
        x = Activation(activation)(x)
    
    return x

def resnet_block(input, filters:list, kernel_size:tuple=(3,3), strides:tuple=(1,1), convolutional:bool=False):
    """ Performs either identity block or convolutional block.
    
    Inputs
    ------------
    input: An input List of filters. e.g [64, 64, 256]
    kernel_size (tuple): Size of each kernel
    strides (tuple): Strides for tranlating the kernel
    convolutional (bool): If True, then convolutional block, otheriwise identity block.
    
    Outputs
    --------------
    x: An output tensor
    
    """
    x_shortcut = input
    
    x = conv_block(input, filters[0], (1,1), strides, 'valid', 3, 'relu')
    x = conv_block(x, filters[1], kernel_size, (1,1), 'same', 3, 'relu')    
    x = conv_block(x, filters[2], (1,1), (1,1), 'valid', 3, None) 
    
    if convolutional: x_shortcut = conv_block(input, filters[2], (1,1), strides, 'valid', 3, None)
        
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x

def resnet50(input_shape, n_classes, aug=None):
    """ Implements ResNet50 architecture
    
    Inputs
    ------------
    input_shape: Shape of the input image. e.g (32, 32, 3)
    n_classes: No. of classes
    aug: Any augmentation can be passed through it.
    
    Outputs
    ------------
    model: ResNet50 model
    """
    
    input = Input(input_shape)
    
    # Stage 0: zero-padding
    if aug is not None: # with augmentation
      x = aug(input)
      x = ZeroPadding2D((3, 3))(x) 
    else: x = ZeroPadding2D((3, 3))(input) # without augmentation
    
    # Stage 1
    x = conv_block(x, 64, (7,7), (2,2), 'valid', 3, 'relu')    
    x = MaxPooling2D((3,3), strides=(2,2))(x)
    
    # Stage 2
    x = resnet_block(x, [64,64,256], (3,3), (1,1), True)
    x = resnet_block(x, [64,64,256], (3,3), (1,1), False)
    x = resnet_block(x, [64,64,256], (3,3), (1,1), False)
    
    # Stage 3
    x = resnet_block(x, [128,128,512], (3,3), (2,2), True)
    x = resnet_block(x, [128,128,512], (3,3), (1,1), False)
    x = resnet_block(x, [128,128,512], (3,3), (1,1), False)
    x = resnet_block(x, [128,128,512], (3,3), (1,1), False)
    
    # Stage 4
    x = resnet_block(x, [256,256,1024], (3,3), (2,2), True)
    x = resnet_block(x, [256,256,1024], (3,3), (1,1), False)
    x = resnet_block(x, [256,256,1024], (3,3), (1,1), False)
    x = resnet_block(x, [256,256,1024], (3,3), (1,1), False)
    x = resnet_block(x, [256,256,1024], (3,3), (1,1), False)
    x = resnet_block(x, [256,256,1024], (3,3), (1,1), False)
    
    # Stage 5
    x = resnet_block(x, [512,512,2048], (3,3), (2,2), True)
    x = resnet_block(x, [512,512,2048], (3,3), (1,1), False) 
    x = resnet_block(x, [512,512,2048], (3,3), (1,1), False) 
    
    # Stage 6: Average pooling
    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    
    # Output layer
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax', name='fc')(x)

    # Create model
    model = Model(inputs=input, outputs=x, name='ResNet50')

    return model


if __name__ == "__main__":    
    model = resnet50(input_shape=(64, 64, 3), n_classes=6)    
    
    model.summary()
