import keras
from model_parameters import SIZE

def basic_block(inputs, channels, strides = 1):
    factor = 6
    x = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding='same',
                                     use_bias=False, kernel_initializer='he_normal')(inputs)
    x = keras.layers.Conv2D(channels//factor, kernel_size=(1,1), strides=1,kernel_regularizer=l2(0.00004),
                            use_bias=False, kernel_initializer='he_normal')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same',
                                     use_bias=False, kernel_initializer='he_normal')(x)
    x = keras.layers.Conv2D(channels, kernel_size=(1,1), strides=1,kernel_regularizer=l2(0.00004),
                            use_bias=False, kernel_initializer='he_normal')(x)

    if strides == 2:
        inputs = keras.layers.Conv2D(channels, kernel_size=(1,1), strides=strides,kernel_regularizer=l2(0.00004),
                                     use_bias=False, kernel_initializer='he_normal')(inputs)

    if inputs.get_shape()[3] != channels:
        inputs = keras.layers.Conv2D(channels, kernel_size=(1,1), strides=strides,kernel_regularizer=l2(0.00004),
                                     use_bias=False, kernel_initializer='he_normal')(inputs)

    x = keras.layers.Add()([inputs,x])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x

def updown_block(inputs,channels=8,strides=2,decoder=False):
    x = keras.layers.Conv2D(channels,kernel_size=(3,3),strides=strides,
                            kernel_initializer='he_normal',padding='same')(inputs)
    if decoder:
        x = keras.layers.Conv2DTranspose(channels,kernel_size=(3,3),strides=2,kernel_regularizer=l2(0.00004),
                                         kernel_initializer='he_normal',padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x

def JPU(inputs, channels = 128):
    conv3 = keras.layers.Conv2D(32,kernel_size=(1,1),strides=(1,1),use_bias=False,
                                kernel_regularizer=l2(0.00004),kernel_initializer='he_normal')(inputs[0])
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    conv4 = keras.layers.Conv2D(32,kernel_size=(1,1),strides=(1,1),use_bias=False,
                                kernel_regularizer=l2(0.00004), kernel_initializer='he_normal')(inputs[1])
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Activation('relu')(conv4)
    conv4 = keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear')(conv4)

    conv5 = keras.layers.Conv2D(32,kernel_size=(1,1),strides=(1,1),use_bias=False,
                                kernel_regularizer=l2(0.00004), kernel_initializer='he_normal')(inputs[2])
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Activation('relu')(conv5)
    conv5 = keras.layers.UpSampling2D(size=(4,4),interpolation='bilinear')(conv5)

    concat = keras.layers.Concatenate()([conv3,conv4,conv5])
    concat = keras.layers.Conv2D(32,kernel_size=(1,1),strides=(1,1),use_bias=False,
                                 kernel_regularizer=l2(0.00004), kernel_initializer='he_normal')(concat)
    concat = keras.layers.BatchNormalization()(concat)
    concat = keras.layers.Activation('relu')(concat)


    s1 = keras.layers.DepthwiseConv2D(kernel_size=(3,3),dilation_rate=1,padding='same',
                                      use_bias=False, kernel_initializer='he_normal')(concat)
    #s1 = keras.layers.Conv2D(32,kernel_size=(1,1),use_bias=False, kernel_initializer='he_normal')(s1)
    s1 = keras.layers.BatchNormalization()(s1)
    s1 = keras.layers.Activation('relu')(s1)

    s2 = keras.layers.DepthwiseConv2D(kernel_size=(3,3),dilation_rate=2,padding='same',
                                      use_bias=False, kernel_initializer='he_normal')(concat)
    #s2 = keras.layers.Conv2D(32,kernel_size=(1,1),use_bias=False, kernel_initializer='he_normal')(s2)
    s2 = keras.layers.BatchNormalization()(s2)
    s2 = keras.layers.Activation('relu')(s2)

    s3 = keras.layers.DepthwiseConv2D(kernel_size=(3,3),dilation_rate=4,padding='same',
                                      use_bias=False, kernel_initializer='he_normal')(concat)
    #s3 = keras.layers.Conv2D(32,kernel_size=(1,1),use_bias=False, kernel_initializer='he_normal')(s3)
    s3 = keras.layers.BatchNormalization()(s3)
    s3 = keras.layers.Activation('relu')(s3)

    s4 = keras.layers.DepthwiseConv2D(kernel_size=(3,3),dilation_rate=8,padding='same',
                                      use_bias=False, kernel_initializer='he_normal')(concat)
    #s4 = keras.layers.Conv2D(32,kernel_size=(1,1),use_bias=False, kernel_initializer='he_normal')(s4)
    s4 = keras.layers.BatchNormalization()(s4)
    s4 = keras.layers.Activation('relu')(s4)

    concat = keras.layers.Concatenate()([s1,s2,s3, s4])

    concat = keras.layers.Conv2D(channels,kernel_size=(1,1),strides=(1,1),use_bias=False,
                                 kernel_regularizer=l2(0.00004),kernel_initializer='he_normal')(concat)
    concat = keras.layers.BatchNormalization()(concat)
    concat = keras.layers.Activation('relu')(concat)

    return concat

def architecture(input_size=(SIZE[0],SIZE[1],3),classes=19, depths = [6, 16, 8],channels = [64, 128, 256]):
    input_layer = keras.layers.Input(input_size)

    stg1 = keras.layers.Conv2D(16,kernel_size=(3,3),strides=(2,2),padding='same',kernel_regularizer=l2(0.00004),
                               use_bias=False, kernel_initializer='he_normal')(input_layer)
    stg1 = keras.layers.BatchNormalization()(stg1)
    stg1 = keras.layers.Activation('relu')(stg1)

    stg2 = keras.layers.Conv2D(32,kernel_size=(3,3),strides=(2,2),padding='same',kernel_regularizer=l2(0.00004),
                               use_bias=False, kernel_initializer='he_normal')(stg1)
    stg2 = keras.layers.BatchNormalization()(stg2)
    stg2 = keras.layers.Activation('relu')(stg2)

    for i in range(depths[0]):
        if i == 0:
            stg3 = basic_block(stg2, channels[0], strides = 2)
        else:
            stg3 = basic_block(stg3, channels[0], strides = 1)

    for i in range(depths[1]):
        if i == 0:
            stg4 = basic_block(stg3, channels[1], strides = 2)
        else:
            stg4 = basic_block(stg4, channels[1], strides = 1)

    for i in range(depths[2]):
        if i == 0:
            stg5 = basic_block(stg4, channels[2], strides = 2)
        else:
            stg5 = basic_block(stg5, channels[2], strides = 1)

    stg5 = JPU([stg3, stg4, stg5])
    x = updown_block(stg5,channels=32,decoder=True)

    x = keras.layers.Concatenate()([x, stg2])
    x = updown_block(x,channels=16,decoder=True)

    x = keras.layers.Concatenate()([x, stg1])

    x = updown_block(x,channels=19,decoder=True)
    x = keras.layers.Dropout(0.3)(x)
    #x = keras.layers.Reshape((input_size[0]*input_size[1], -1))(x)
    x = keras.layers.Activation('softmax')(x)

    model = keras.models.Model(input_layer,x)
    return model
