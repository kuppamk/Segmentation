import keras
from keras.regularizers import l2


def fireblock(inputs , channels):
    x = keras.layers.Conv2D(channels[0], kernel_size=(1,1), strides=1, use_bias=False,kernel_regularizer=l2(0.00004),
                            kernel_initializer='he_normal')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    point = keras.layers.Conv2D(channels[1], kernel_size=(1,1), strides=1, use_bias=False,kernel_regularizer=l2(0.00004),
                                kernel_initializer='he_normal')(x)
    point = keras.layers.BatchNormalization()(point)
    point = keras.layers.Activation('relu')(point)

    conv = keras.layers.Conv2D(channels[2], kernel_size=(3,3), strides=1, use_bias=False,kernel_regularizer=l2(0.00004),
                               padding='same',kernel_initializer='he_normal')(x)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Activation('relu')(conv)

    out = keras.layers.Concatenate()([conv,point])
    x = keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same',
                                     use_bias=False, kernel_initializer='he_normal')(out)
    x = keras.layers.Conv2D(channels[2], kernel_size=(1,1), strides=1,
                            use_bias=False, kernel_initializer='he_normal')(x)
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

def architecture(input_size=(512,1024,3),classes=19):
    input_layer = keras.layers.Input(input_size)

    stg1 = keras.layers.Conv2D(57, kernel_size=(3,3), strides=2, use_bias=False, kernel_regularizer=l2(0.00004),
                               padding='same',kernel_initializer='he_normal')(input_layer)
    stg1 = keras.layers.BatchNormalization()(stg1)
    stg1 = keras.layers.Activation('relu')(stg1)

    stg2 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(stg1)
    stg2 = fireblock(stg2,[15,49,53])
    stg2 = fireblock(stg2, [15, 54, 52])

    stg3 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(stg2)
    stg3 = fireblock(stg3,[29, 92,94])
    stg3 = fireblock(stg3, [29, 90, 83])

    stg4 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(stg3)
    stg4 = fireblock(stg4, [44, 166, 161])
    stg4 = fireblock(stg4, [45, 155, 171])
    stg4 = fireblock(stg4, [49, 163, 171])
    stg4 = fireblock(stg4, [25, 29, 54])

    stg5 = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(stg4)
    stg5 = fireblock(stg5,[37, 45, 56])
    stg5 = fireblock(stg5,[37, 45, 56])
    stg5 = fireblock(stg5,[37, 45, 56])
    stg5 = fireblock(stg5,[37, 45, 56])

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

