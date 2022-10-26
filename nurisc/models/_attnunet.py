
import tensorflow as tf
from keras import models, layers, regularizers
from keras import backend as K
import  numpy as np
from keras.layers import Input
from keras.models import Model




def expend_as(tensor, rep):
     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def double_conv_layer(x, filter_size, size, dropout, batch_norm=False):


    axis = 3
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=axis)(conv)
    conv = layers.Activation('relu')(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=axis)(shortcut)

    res_path = layers.add([shortcut, conv])
    return res_path

def gating_signal(input, out_size, batch_norm=False):

    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    if x is not None:
     shape_x = K.int_shape(x)
    else:
        shape_x=[2,2]

    if gating is not None:
        shape_g = K.int_shape(gating)
        print(shape_g)
    else:
         shape_g  = (2,2)
    # print(shape_g[1])
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    if theta_x is not None:
        shape_theta_x = K.int_shape(theta_x)
    else:
        shape_theta_x=[2,2]

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 # strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                   strides=(2,2),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)

    upsample_psi = layers.UpSampling2D(
        # size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
        size=(2,2))(sigmoid_xg)

    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn


def Attention_ResUNet(n_filter_base=16, batch_norm=True):


    axis = 3
    dropout_rate = 0.0
    def f(input):
        inputs = layers.Input(input().shape, dtype=tf.float32)
        conv_128 = double_conv_layer(inputs, 3, n_filter_base, dropout_rate, batch_norm)
        pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
        # DownRes 2
        conv_64 = double_conv_layer(pool_64, 3, 2*n_filter_base, dropout_rate, batch_norm)
        pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
        # DownRes 3
        conv_32 = double_conv_layer(pool_32, 3, 4*n_filter_base, dropout_rate, batch_norm)
        pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
        # DownRes 4
        conv_16 = double_conv_layer(pool_16, 3, 8*n_filter_base, dropout_rate, batch_norm)
        pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
        # DownRes 5, convolution only
        conv_8 = double_conv_layer(pool_8, 3, 16*n_filter_base, dropout_rate, batch_norm)

        # Upsampling layers
        # UpRes 6, attention gated concatenation + upsampling + double residual convolution
        gating_16 = gating_signal(conv_8, 8*n_filter_base, batch_norm)
        att_16 = attention_block(conv_16, gating_16, 8*n_filter_base)
        up_16 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(conv_8)
        up_16 = layers.concatenate([up_16, att_16], axis=axis)
        up_conv_16 = double_conv_layer(up_16, 3, 8*n_filter_base, dropout_rate, batch_norm)
        # UpRes 7
        gating_32 = gating_signal(up_conv_16, 4*n_filter_base, batch_norm)
        att_32 = attention_block(conv_32, gating_32, 4*n_filter_base)
        up_32 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_16)
        up_32 = layers.concatenate([up_32, att_32], axis=axis)
        up_conv_32 = double_conv_layer(up_32, 3, 4*n_filter_base, dropout_rate, batch_norm)
        # UpRes 8
        gating_64 = gating_signal(up_conv_32, 2*n_filter_base, batch_norm)
        att_64 = attention_block(conv_64, gating_64, 2*32)
        up_64 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_32)
        up_64 = layers.concatenate([up_64, att_64], axis=axis)
        up_conv_64 = double_conv_layer(up_64, 3, 2*n_filter_base, dropout_rate, batch_norm)
        # UpRes 9
        gating_128 = gating_signal(up_conv_64, n_filter_base, batch_norm)
        att_128 = attention_block(conv_128, gating_128, 32)
        up_128 = layers.UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_64)
        up_128 = layers.concatenate([up_128, att_128], axis=axis)
        up_conv_128 = double_conv_layer(up_128, 3, n_filter_base, dropout_rate, batch_norm)
        # 1*1 convolutional layers
        # valid padding
        # batch normalization
        # sigmoid nonlinear activation
        conv_final = layers.Conv2D(1, kernel_size=(1,1))(up_conv_128)
        conv_final = layers.BatchNormalization(axis=axis)(conv_final)
        conv_final = layers.Activation('relu')(conv_final)

        # Model integration
        # model = models.Model(inputs, conv_final, name="AttentionResUNet")
        return conv_final
    return f


# convolutional block
def conv_block(x, kernelsize, filters, dropout, batchnorm=False):
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(x)
    if batchnorm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)
    conv = layers.Conv2D(filters, (kernelsize, kernelsize), kernel_initializer='he_normal', padding="same")(conv)
    if batchnorm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    return conv


# gating signal for attention unit
def gatingsignal(input, out_size, batchnorm=False):
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


# attention unit/block based on soft attention
# def attention_block(x, gating, inter_shape):
#     shape_x = K.int_shape(x)
#     shape_g = K.int_shape(gating)
#     theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x)
#     shape_theta_x = K.int_shape(theta_x)
#     phi_g = layers.Conv2D(inter_shape, (1, 1), kernel_initializer='he_normal', padding='same')(gating)
#     upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
#                                         strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
#                                         kernel_initializer='he_normal', padding='same')(phi_g)
#     concat_xg = layers.add([upsample_g, theta_x])
#     act_xg = layers.Activation('relu')(concat_xg)
#     psi = layers.Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
#     sigmoid_xg = layers.Activation('sigmoid')(psi)
#     shape_sigmoid = K.int_shape(sigmoid_xg)
#     upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(
#         sigmoid_xg)
#     upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
#                                  arguments={'repnum': shape_x[3]})(upsample_psi)
#     y = layers.multiply([upsample_psi, x])
#     result = layers.Conv2D(shape_x[3], (1, 1), kernel_initializer='he_normal', padding='same')(y)
#     attenblock = layers.BatchNormalization()(result)
#     return attenblock
def attentionunet(inputs, batchnorm=True):
    filters = [16, 32, 64, 128, 256]
    kernelsize = 3
    upsample_size = 2
    dropout = 0.2
    # inputs = layers.Input(input_shape)

    # Downsampling layers
    dn_1 = conv_block(inputs, kernelsize, filters[0], dropout, batchnorm)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(dn_1)

    dn_2 = conv_block(pool_1, kernelsize, filters[1], dropout, batchnorm)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(dn_2)

    dn_3 = conv_block(pool_2, kernelsize, filters[2], dropout, batchnorm)
    pool_3 = layers.MaxPooling2D(pool_size=(2, 2))(dn_3)

    dn_4 = conv_block(pool_3, kernelsize, filters[3], dropout, batchnorm)
    pool_4 = layers.MaxPooling2D(pool_size=(2, 2))(dn_4)

    dn_5 = conv_block(pool_4, kernelsize, filters[4], dropout, batchnorm)

    # Upsampling layers
    gating_5 = gatingsignal(dn_5, filters[3], batchnorm)
    att_5 = attention_block(dn_4, gating_5, filters[3])
    up_5 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(dn_5)
    up_5 = layers.concatenate([up_5, att_5], axis=3)
    up_conv_5 = conv_block(up_5, kernelsize, filters[3], dropout, batchnorm)

    gating_4 = gatingsignal(up_conv_5, filters[2], batchnorm)
    att_4 = attention_block(dn_3, gating_4, filters[2])
    up_4 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_5)
    up_4 = layers.concatenate([up_4, att_4], axis=3)
    up_conv_4 = conv_block(up_4, kernelsize, filters[2], dropout, batchnorm)

    gating_3 = gatingsignal(up_conv_4, filters[1], batchnorm)
    att_3 = attention_block(dn_2, gating_3, filters[1])
    up_3 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_4)
    up_3 = layers.concatenate([up_3, att_3], axis=3)
    up_conv_3 = conv_block(up_3, kernelsize, filters[1], dropout, batchnorm)

    gating_2 = gatingsignal(up_conv_3, filters[0], batchnorm)
    att_2 = attention_block(dn_1, gating_2, filters[0])
    up_2 = layers.UpSampling2D(size=(upsample_size, upsample_size), data_format="channels_last")(up_conv_3)
    up_2 = layers.concatenate([up_2, att_2], axis=3)
    up_conv_2 = conv_block(up_2, kernelsize, filters[0], dropout, batchnorm)

    conv_final = layers.Conv2D(1, kernel_size=(1, 1))(up_conv_2)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    outputs = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model


if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)

    inp = Input((256, 256, 1))

    features = Attention_ResUNet(n_filter_base=32, batch_norm=True)(inp)

    out = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(features)

    model = Model(inp, out)

    model.summary()