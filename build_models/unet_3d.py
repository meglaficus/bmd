from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, BatchNormalization, Activation, UpSampling3D, MaxPooling3D, Conv3D


#################
## Build U-Net ##
#################

def build_unet(input_shape):

    inputs = Input(input_shape)

    cur_f = 16
    prev = inputs
    encs = []

    for i in range(4):
        e = basic_block(cur_f, 3, 1, prev)
        e = basic_block(cur_f, 3, 1, e)

        encs.append(e)

        if i < 2:
            cur_f *= 2

        e = MaxPooling3D(pool_size=(2, 2, 2))(e)
        prev = e

    prev = basic_block(cur_f, 3, 1, prev)
    prev = basic_block(cur_f, 3, 1, prev)

    num_of_encs = len(encs)

    for i in range(num_of_encs):
        if num_of_encs - i <= 2:
            cur_f = int(cur_f / 2)

        enc = encs[num_of_encs - 1 - i]

        e = UpSampling3D(size=(2, 2, 2))(prev)
        e = concatenate([e, enc], axis=-1)
        e = basic_block(cur_f, 3, 1, e)
        e = basic_block(cur_f, 3, 1, e)
        prev = e

    dec = Conv3D(1, 1, strides=1, padding="same", kernel_initializer='he_uniform')(prev)
    dec = Activation(activation='sigmoid')(dec)

    m = Model(inputs=inputs, outputs=dec)
    return m


def basic_block(F, K, S, inputs):
    enc1 = Conv3D(F, K, strides=S, padding="same", kernel_initializer='he_uniform')(inputs)
    enc1 = BatchNormalization()(enc1)
    enc1 = Activation(activation='relu')(enc1)
    return enc1

