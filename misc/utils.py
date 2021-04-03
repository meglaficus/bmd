from keras import backend as K
import numpy as np
import os
import sys
import array
import pydicom


class MyClass:
    def __init__(self):
        pass


def read_dicom(dir_path):

    dcms = []
    for d, s, fl in os.walk(dir_path):
        for fn in fl:
            if ".dcm" in fn.lower():
                dcms.append(os.path.join(d, fn))
    dcms.sort()
    ref_dicom = pydicom.read_file(dcms[0])

    d_array = np.zeros((len(dcms), ref_dicom.Rows, ref_dicom.Columns), dtype=ref_dicom.pixel_array.dtype)

    loc = []
    for dcm in dcms:
        d = pydicom.read_file(dcm)
        loc.append(float(d.SliceLocation))
    loc.sort()

    for dcm in dcms:
        d = pydicom.read_file(dcm)
        index = loc.index(float(d.SliceLocation))
        d_array[index, :, :] = d.pixel_array

    return d_array


def read_binary_sequence(file_full_path):

    file_size = os.stat(file_full_path).st_size  # returns byte size
    number_of_pixels = int(file_size)

    file = open(file_full_path, 'rb')
    image_value = array.array('B')
    image_value.fromfile(file, number_of_pixels)
    file.close()

    if sys.byteorder == 'little':
        image_value.byteswap()

    return image_value


def read_raw(input_file_path):

    pixel_values = read_binary_sequence(input_file_path)
    a = np.array(pixel_values)
    a.shape = (-1, 512, 512)
    aa = a.astype(np.uint16)

    return aa


def window_setting_for_ct(im, w=1000, l=300):

    window_top = l + w / 2
    window_bottom = l - w / 2

    im = np.clip(im, window_bottom, window_top)
    im = ((im - window_bottom) * 255 / w).astype(np.uint8)

    return im


def grayscale_to_rgb(im255):

    im_rgb = np.concatenate([im255, im255, im255], axis=3).astype("float16")

    return im_rgb


def dice_coef(y_true, y_pred):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)

    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):

    return 1.0 - dice_coef(y_true, y_pred)


def make_meshgrid():

    a = np.arange(0, 480, 32)

    x, y = np.meshgrid(a, a)

    x = y.ravel()
    y = y.ravel()

    return x, y


def TP(y_true, y_pred):

    return K.sum((y_true[:, 1] * K.round(y_pred[:, 1])))


def TN(y_true, y_pred):

    return K.sum(y_true[:, 0] * K.round(y_pred[:, 0]))


def FP(y_true, y_pred):

    return K.sum(K.cast(K.equal(K.round(y_pred[:, 1]) - y_true[:, 1], 1), K.floatx()))


def FN(y_true, y_pred):

    return K.sum(K.cast(K.equal(K.round(y_pred[:, 0]) - y_true[:, 0], 1), K.floatx()))


def pre(y_true, y_pred):

    return (TP(y_true, y_pred)) / (TP(y_true, y_pred) + FP(y_true, y_pred) + K.epsilon())

def rec(y_true, y_pred):

    return (TP(y_true, y_pred)) / (TP(y_true, y_pred) + FN(y_true, y_pred) + K.epsilon())


def F(y_true, y_pred):

    return (pre(y_true, y_pred) * rec(y_true, y_pred) * 2) / (pre(y_true, y_pred) + rec(y_true, y_pred) + K.epsilon())


