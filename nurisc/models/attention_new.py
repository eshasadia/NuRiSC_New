
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    add, multiply
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K

from csbdeep.utils import _raise, backend_channels_last
import tensorflow as tf
import numpy as np


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x

def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate








def att_unet( n_features = 64, data_format='channels_first'):
    # inputs = Input((3, img_w, img_h))
    # x = inputs

   def f(x):
        depth = 4
        n_label = 65
        skips = []

        for i in range(depth):
            x = Conv2D(n_features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
            x = Dropout(0.2)(x)
            x = Conv2D(n_features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
            skips.append(x)
            x = MaxPooling2D((2, 2), data_format='channels_first')(x)
            n_features = n_features * 2

        x = Conv2D(n_features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(n_features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

        for i in reversed(range(depth)):
            features = features // 2
            x = attention_up_and_concate(x, skips[i], data_format=data_format)
            x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
            x = Dropout(0.2)(x)
            x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

        conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
        conv7 = core.Activation('sigmoid')(conv6)
        return conv7


    # model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
   return f


if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)

    inp = Input((None, None, 1))

    features = att_unet(features=2048)(inp)

    out = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(features)

    model = Model(inp, out)

    model.summary()
