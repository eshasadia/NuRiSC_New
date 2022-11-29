##our method
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation,Reshape, \
                        BatchNormalization, PReLU, Deconvolution3D,Add,SpatialDropout3D,\
                            add,GlobalAveragePooling3D,AveragePooling3D,multiply,Lambda,Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils import plot_model
#from keras.utils import multi_gpu_model

from metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient,weighted_dice_coefficient_loss

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate



def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

# def resudial_block(input_layer, n_filters,kernel=(3, 3, 3), padding='same', strides=(1, 1, 1)):
#     layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
#     layer = BatchNormalization(axis=1)(layer)
#     layer = Activation("relu")(layer)
#     layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer)
#     layer = BatchNormalization(axis=1)(layer)
#     x_short = Conv3D(n_filters,kernel,padding=padding,strides=strides)(input_layer)
#     x_short = BatchNormalization(axis=1)(x_short)
#     # print("layer shape",layer.shape)
#     # print("input layer shape", input_layer.shape)
#     layer = Add()([x_short, layer])
#     return Activation("relu")(layer)
#
# def resudial_block_2(input_layer, n_filters,kernel=(3, 3, 3), padding='same', strides=(1, 1, 1)):
#     layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
#     layer = BatchNormalization(axis=1)(layer)
#     layer = Activation("relu")(layer)
#     layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer)
#     layer = BatchNormalization(axis=1)(layer)
#     # print("layer shape",layer.shape)
#     # print("input layer shape", input_layer.shape)
#     layer = Add()([input_layer, layer])
#     return Activation("relu")(layer)

# def attention_block_3d(x, g, inter_channel):
#     '''
#     :param x: x input from down_sampling same layer output x(?,x_height,x_width,x_depth,x_channel)
#     :param g: gate input from up_sampling layer last output g(?,g_height,g_width,g_depth,g_channel)
#     g_height,g_width,g_depth=x_height/2,x_width/2,x_depth/2
#     :return:
#     '''
#     # theta_x(?,g_height,g_width,g_depth,inter_channel)
#     theta_x = Conv3D(inter_channel, [2, 2, 2], strides=[2, 2, 2])(x)
#
#     # phi_g(?,g_height,g_width,g_depth,inter_channel)
#     phi_g = Conv3D(inter_channel, [1, 1, 1], strides=[1, 1, 1])(g)
#
#     # f(?,g_height,g_width,g_depth,inter_channel)
#     f = Activation('relu')(keras.layers.add([theta_x, phi_g]))
#
#     # psi_f(?,g_height,g_width,g_depth,1)
#     psi_f = Conv3D(1, [1, 1, 1], strides=[1, 1, 1])(f)
#
#     # sigm_psi_f(?,g_height,g_width,g_depth)
#     sigm_psi_f = Activation('sigmoid')(psi_f)
#
#     # rate(?,x_height,x_width,x_depth)
#     # rate = UpSampling3D(size=[2, 2, 2])(sigm_psi_f)
#
#     # att_x(?,x_height,x_width,x_depth,x_channel)
#     att_x = keras.layers.multiply([x, rate])
#
#     return att_x
def expand_dim_backend(x):
    x = K.expand_dims(x,-1)
    x = K.expand_dims(x,-1)
    x = K.expand_dims(x,-1)
    return x
def senet(layer, n_filter):
    seout = GlobalAveragePooling3D()(layer)
    seout = Dense(units=int(n_filter/2))(seout)
    seout = Activation("relu")(seout)
    seout = Dense(units=n_filter)(seout)
    seout = Activation("sigmoid")(seout)
    print("seout1 shape",seout.shape)
    # seout = Reshape([-1,1,1,n_filter])(seout)
    seout = Lambda(expand_dim_backend)(seout)
    print("seout shape",seout.shape)
    return seout

def resudial_block(input_layer, n_filters,kernel_1=(1, 1, 1),kernel_3=(3,3,3), padding='same', strides=(1, 1, 1)):
    layer = BatchNormalization(axis=1)(input_layer)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(layer)

    layer_1 = BatchNormalization(axis=1)(layer)
    layer_1 = Activation("relu")(layer_1)
    layer_1 = Conv3D(int(n_filters/2), kernel_3, padding=padding, strides=strides)(layer_1)

    layer_2 = BatchNormalization(axis=1)(layer)
    layer_2 = Activation('relu')(layer_2)
    layer_2 = Conv3D(int(n_filters/2),(3,3,1), padding=padding, strides=strides)(layer_2)

    layer_2_1 = BatchNormalization(axis=1)(layer_2)
    layer_2_1 = Activation('relu')(layer_2_1)
    layer_2_1 = Conv3D(int(n_filters/4), (1, 3, 3), padding=padding, strides=strides)(layer_2_1)
    layer_2_2 = BatchNormalization(axis=1)(layer_2)
    layer_2_2 = Activation('relu')(layer_2_2)
    layer_2_2 = Conv3D(int(n_filters/4), (3, 1, 3), padding=padding, strides=strides)(layer_2_2)

    layer = concatenate([layer_1,layer_2,layer_2_1,layer_2_2],axis=1)

    seout = senet(layer,int(n_filters/2*3))
    seout = multiply([seout,layer])

    layer = BatchNormalization(axis=1)(seout)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(layer)

    x_short = Conv3D(n_filters,kernel_1,padding=padding,strides=strides)(input_layer)

    layer_out = add([x_short, layer])

    return layer_out

def resudial_block_2(input_layer, n_filters,kernel_1=(1, 1, 1),kernel_3=(3,3,3), padding='same', strides=(1, 1, 1)):
    layer = BatchNormalization(axis=1)(input_layer)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(layer)

    layer_1 = BatchNormalization(axis=1)(layer)
    layer_1 = Activation("relu")(layer_1)
    layer_1 = Conv3D(n_filters, kernel_3, padding=padding, strides=strides)(layer_1)

    layer = BatchNormalization(axis=1)(layer_1)
    layer = Activation("relu")(layer)
    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(layer)

    x_short = input_layer

    layer = add([x_short, layer])
    return layer

def unet_model_3d(input_shape, pool_size=(2, 2, 1), n_labels=1, initial_learning_rate=0.0001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=True, activation_name="sigmoid"):
    ############################
    #resUnet + dropout
    ###############################
    inputs = input_shape
    # current_layer = inputs    
    inputs_1 = Conv3D(n_base_filters,kernel_size=(3,3,3),strides=(1,1,1),padding="same")(inputs)
    inputs_1 = BatchNormalization(axis=1)(inputs_1)
    inputs_1 = Activation("relu")(inputs_1)
    layer1 = resudial_block(inputs_1,n_base_filters)
    #layer1 = resudial_block_2(layer1,n_base_filters)
    layer1_pool = MaxPooling3D(pool_size=(2,2,2))(layer1)
    print(layer1_pool.shape)
    layer2 = resudial_block(layer1_pool,n_base_filters*2)
    #layer2 = resudial_block_2(layer2,n_filters=n_base_filters*2)
    layer2_poo2 = MaxPooling3D(pool_size=pool_size)(layer2)
    print(layer1_pool.shape)
    layer3 = resudial_block(layer2_poo2,n_base_filters*4)
    #layer3 = resudial_block_2(layer3,n_base_filters*4)
    layer3_poo3 = MaxPooling3D(pool_size=pool_size)(layer3)
    print(layer1_pool.shape)
    layer3_poo3 = SpatialDropout3D(rate=0.1)(layer3_poo3)
    layer4 = Conv3D(n_base_filters*8, kernel_size=(3,3,3), padding="same", strides=(1,1,1))(layer3_poo3)
    layer4 = BatchNormalization(axis=1)(layer4)
    layer4 = Activation("relu")(layer4)
    layer4 = Conv3D(n_base_filters * 8, kernel_size=(3, 3, 3), padding="same", strides=(1, 1, 1))(layer4)
    layer4 = BatchNormalization(axis=1)(layer4)
    layer4 = Activation("relu")(layer4)
    layer4 = SpatialDropout3D(rate=0.1)(layer4)

    layer_up_3 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                            n_filters=n_base_filters*3)(layer4)
    print("layer3:",layer3.shape)
    print("layer_up_3:",layer_up_3.shape)
    concat3 = concatenate([layer_up_3, layer3], axis=1)
    print("concat3:", concat3.shape)
    layer33 = resudial_block(concat3,n_base_filters*4)
    #layer33 = resudial_block_2(layer33,n_base_filters*4)

    layer_up_2 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                    n_filters=n_base_filters * 2)(layer33)
    concat2 = concatenate([layer_up_2, layer2], axis=1)
    layer22 = resudial_block(concat2, n_base_filters * 2)
    #layer22 = resudial_block_2(layer22, n_base_filters * 2)

    layer_up_1 = get_up_convolution(pool_size=(2,2,2), deconvolution=False,
                                    n_filters=n_base_filters * 1)(layer22)
    concat1 = concatenate([layer_up_1, layer1], axis=1)
    layer11 = resudial_block(concat1, n_base_filters * 1)
    #layer11 = resudial_block_2(layer11, n_base_filters * 1)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(layer11)
    print("final_convolution.shape:",final_convolution.shape)
    act = Activation(activation_name)(final_convolution)
    print("act.shape:", act.shape)
    model = Model(inputs=inputs, outputs=act)

    #plot_model(model, to_file='model.png')
    if not isinstance(metrics, list):
        metrics = [metrics]

    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer.lr
    #     return lr
    # lr_metric = get_lr_metric(Adam(lr=initial_learning_rate))

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    print(model.summary())
    return model
