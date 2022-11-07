# Import Necessary Libraries
import numpy as np
import tensorflow as tf
from keras.layers import Input, Activation,BatchNormalization,add, Conv2DTranspose, Conv2D, concatenate, Add, MaxPooling2D,UpSampling2D
from keras.models import Model
def Conv_Block(inputs, model_width, num_row, num_col, activation='relu',  padding='same',batch_norm=False, name=None):
    # 2D Convolutional Block
    x = Conv2D(model_width , (num_row,num_col), padding=padding)(inputs)
    if batch_norm:
        x = BatchNormalization(axis=-1)(x)

    if not activation is None:
        x = Activation(activation, name=name)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)

    return x


def trans_conv2D(inputs, model_width, activation=None, strides=(2, 2), batch_norm=False, name=None):
    # 2D Transposed Convolutional Block, used instead of UpSampling
    x = Conv2DTranspose(model_width , (2, 2), strides=strides, padding='same')(inputs)  # Stride = 2, Kernel Size = 2
    if batch_norm:
        x = BatchNormalization(axis=-1)(x)
    x = BatchNormalization()(x)
    if not activation is None:
        x = Activation(activation, name=name)(x)
        # x = tf.keras.layers.Activation(activation,'relu')(x)

    return x


def Concat_Block(input1, *argv):
    # Concatenation Block from the Keras Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs, size=(2, 2)):
    # 2D UpSampling Block
    up = UpSampling2D(size=size)(inputs)

    return up

def Attention_Block(skip_connection, gating_signal, num_filters):
    # Attention Block
    conv1x1_1 = Conv2D(num_filters, (1, 1), strides=(2, 2))(skip_connection)
    conv1x1_1 =BatchNormalization()(conv1x1_1)
    conv1x1_2 = Conv2D(num_filters, (1, 1), strides=(1, 1))(gating_signal)
    conv1x1_2 = BatchNormalization()(conv1x1_2)
    conv1_2 = add([conv1x1_1, conv1x1_2])
    conv1_2 = Activation('relu')(conv1_2)
    conv1_2 = Conv2D(1, (1, 1), strides=(1, 1))(conv1_2)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Activation('sigmoid')(conv1_2)
    resampler1 = upConv_Block(conv1_2)
    resampler2 = trans_conv2D(conv1_2, 1)
    resampler = add([resampler1, resampler2])
    out = skip_connection * resampler

    return out


def MultiResBlock(inputs, model_width, batch_norm=True, alpha=1.67, padding='same', activation='relu'):
    # MultiRes Block
    # U {int} -- Number of filters in a corrsponding UNet stage
    # inp {keras layer} -- input layer

    w = alpha * model_width

    shortcut = inputs
    shortcut = Conv_Block(shortcut, int(w * 0.167) + int(w * 0.333) + int(w * 0.5), 1, 1, activation=None,
                         batch_norm=batch_norm, padding='same')

    conv3x3 = Conv_Block(inputs, int(w * 0.167), 3, 3,activation=activation, padding=padding, batch_norm=batch_norm)
    conv5x5 = Conv_Block(conv3x3, int(w * 0.333), 3, 3,activation=activation, padding=padding, batch_norm=batch_norm)
    conv7x7 = Conv_Block(conv5x5, int(w * 0.5), 3, 3, activation=activation, padding=padding, batch_norm=batch_norm)

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
    out = BatchNormalization()(out)
    out = Add()([shortcut, out])
    out = Activation(activation)(out)
    out = BatchNormalization()(out)

    return out


def ResPath(inputs, model_width, model_depth=5, batch_norm=True, padding='same'):
    # ResPath
    # filters {int} -- [description]
    # length {int} -- length of ResPath
    # inp {keras layer} -- input layer

    shortcut = inputs
    shortcut = Conv_Block(shortcut, model_width, 1, 1,activation=None, padding=padding, batch_norm=batch_norm)

    out = Conv_Block(inputs, model_width, 3, 3,  activation='relu', padding=padding, batch_norm=batch_norm)
    out = Add()([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)

    for _ in range(1, model_depth):
        shortcut = out
        shortcut = Conv_Block(shortcut, model_width, 1, 1,activation=None, padding=padding, batch_norm=batch_norm)

        out = Conv_Block(out, model_width, 3, 3,  activation='relu', padding=padding, batch_norm=batch_norm)
        out = Add()([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

    return out



def attention_mrunet(model_width, model_depth=5 ):

    def f(x):

        mresblocks = {}
        levels = []

        # Encoding
        # inputs = tf.keras.Input((length, width, num_channel))
        D_S = 1
        A_G = 1
        LSTM = 0
        is_transconv= True
        pool = x

        for i in range(1, (model_depth + 1)):
            mresblock = MultiResBlock(pool, model_width, 2 ** (i - 1))
            pool = MaxPooling2D(pool_size=(2, 2))(mresblock)
            mresblocks["mres%s" % i] = ResPath(mresblock, (model_depth - i + 1), model_width, 2 ** (i - 1))

        # if A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            # pool = Feature_Extraction_Block(pool, model_width, feature_number)

        mresblock = MultiResBlock(pool, model_width, 2 ** model_depth)

        # Decoding
        deconv = mresblock
        mresblocks_list = list(mresblocks.values())

        for j in range(0, model_depth):
            skip_connection = mresblocks_list[model_depth - j - 1]
            if A_G == 1:
                skip_connection = Attention_Block(mresblocks_list[model_depth - j - 1], deconv, model_width)
            if D_S == 1:
                level = Conv2D(1, (1, 1), name=f'level{model_depth - j}')(deconv)
                levels.append(level)
            # if is_transconv:
            #     deconv = trans_conv2D(deconv, model_width, 2 ** (model_depth - j - 1))
            # elif not is_transconv:
                deconv = upConv_Block(deconv)
            # if LSTM == 1:
            #     x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(length / (2 ** (model_depth - j - 1))), np.int32(width / (2 ** (model_depth - j - 1))), np.int32(model_width * (2 ** (model_depth - j - 1)))))(skip_connection)
            #     x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(length / (2 ** (model_depth - j - 1))), np.int32(width / (2 ** (model_depth - j - 1))), np.int32(model_width * (2 ** (model_depth - j - 1)))))(deconv)
            #     merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
            #     deconv = tf.keras.layers.ConvLSTM2D(filters=np.int32(model_width * (2 ** (model_depth - j - 2))), kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
            elif LSTM == 0:
                deconv = Concat_Block(deconv, skip_connection)
            deconv = MultiResBlock(deconv, model_width, 2 ** (model_depth - j - 1))
        return deconv
        # # Output
        # outputs = []
        # outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='softmax', name="out")(deconv)
        #
        #
        # model = tf.keras.Model(inputs=[x], outputs=[outputs])
        #
        # if D_S == 1:
        #     levels.append(outputs)
        #     levels.reverse()
        #     model = tf.keras.Model(inputs=[x], outputs=levels)
    return f




if __name__ == '__main__':
    # Configurations
    # length = 224  # Length of the Image (2D Signal)
    # width = 224  # Width of the Image
    model_name = 'MultiResUNet'  # Name of the Model
    model_depth = 5  # Number of Levels in the CNN Model
    model_width = 64  # Width of the Initial Layer, subsequent layers start from here
    # # kernel_size = 3  # Size of the Kernels/Filter
    # # num_channel = 1  # Number of Channels in the Model
    # D_S = 1  # Turn on Deep Supervision
    # A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    # A_G = 1  # Turn on for Guided Attention
    # LSTM = 1  # Turn on for BiConvLSTM
    # problem_type = 'Classification'  # Problem Type: Regression or Classification
    # output_nums = 1  # Number of Classes for Classification Problems, always '1' for Regression Problems
    # is_transconv = True  # True: Transposed Convolution, False: UpSampling
    # '''Only required if the AutoEncoder Mode is turned on'''
    # feature_number = 1024  # Number of Features to be Extracted
    # '''Only required for MultiResUNet'''
    # alpha = 1  # Model Width Expansion Parameter, for MultiResUNet only
    #
    inp = Input((512,512,3))
    features = attention_mrunet( model_width=64, model_depth=5)(inp)

    out = Conv2D(1, (1, 1), padding='same', activation='softmax')(features)

    model = Model(inp, out)

    model.summary()
    # Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    # Model.summary()
