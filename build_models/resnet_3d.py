from keras import backend as K
from keras.layers import Input, Dense, Conv3D, BatchNormalization, Activation, Concatenate, Lambda, AveragePooling3D, GlobalAveragePooling3D, Add
from keras.models import Model


def Conv3D_fixed(x, F, K, S):

    return Conv3D(filters=F, kernel_size=K, strides=S, padding="same", kernel_initializer='he_uniform')(x)


# Single ReLU, Zero padded shortcut
def basic_block(x, i):

    if i < 3:
        f = 32 * (2 ** i)
    else:
        f = 32 * (2 ** 3)

    ######################
    ### the first half ###
    ######################
    if i == 0:
        shortcut_x = Conv3D_fixed(x, F=f, K=1, S=1)
    else:
        half_x = AveragePooling3D()(x)
        zeros = Lambda(lambda xx: K.zeros_like(xx))(half_x)
        shortcut_x = Concatenate()([half_x, zeros])

    x = BatchNormalization()(x)
    if i == 0:
        x = Conv3D_fixed(x, F=f, K=3, S=1)
    else:
        x = Conv3D_fixed(x, F=f, K=3, S=2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D_fixed(x, F=f, K=3, S=1)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut_x])


    #######################
    ### the second half ###
    #######################
    shortcut_x = Lambda(lambda xx: K.identity(xx))(x)

    x = BatchNormalization()(x)
    x = Conv3D_fixed(x, F=f, K=3, S=1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D_fixed(x, F=f, K=3, S=1)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut_x])


    return x


def build_resnet(input_shape):

    image_input = Input(input_shape)

    x = image_input

    for i in range(4):
        x = basic_block(x, i)

    x = GlobalAveragePooling3D()(x)

    x = Dense(units=2)(x)
    x = Activation("softmax")(x)

    model = Model(inputs=image_input, outputs=x)

    return model