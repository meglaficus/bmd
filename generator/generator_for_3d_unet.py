import numpy as np
from keras.utils import Sequence
import random


##########
## Util ##
##########

def window_setting_and_normalize(im, w=2000, l=400):

    if w == 2000 and l == 400:
        m = -15.42
        s = 273.97
    elif w == 600 and l == 150:
        m = 40.95
        s = 138.71
    elif w == 400 and l == 100:
        m = 39
        s = 106
    elif w == 300 and l == 100:
        m = 47
        s = 86

    window_top = l + w / 2
    window_bottom = l - w / 2

    im = np.clip(im, window_bottom, window_top)
    im = (im - m) / s

    return im


###############
## Generator ##
###############

def mix_up(X1, Y1, X2, Y2, alpha):
    assert X1.shape[0] == Y1.shape[0] == X2.shape[0] == Y2.shape[0]
    batch_size = X1.shape[0]
    l = np.random.beta(alpha, alpha, batch_size)
    X_l = l.reshape(batch_size, 1, 1, 1, 1)
    Y_l = l.reshape(batch_size, 1, 1, 1, 1)
    X = X1 * X_l + X2 * (1 - X_l)
    Y = Y1 * Y_l + Y2 * (1 - Y_l)
    return X, Y


def ricap(image_batch, label_batch, beta=0, use_same_random_value_on_batch=True):

    # if use_same_random_value_on_batch = True : same as the original paper
    assert image_batch.shape[0] == label_batch.shape[0]
    assert image_batch.ndim == 5
    batch_size, image_z, image_y, image_x = image_batch.shape[:4]

    # crop_size w, h ,d from beta distribution
    if use_same_random_value_on_batch:
        w_dash = np.random.beta(beta, beta) * np.ones(batch_size)
        h_dash = np.random.beta(beta, beta) * np.ones(batch_size)
        d_dash = np.random.beta(beta, beta) * np.ones(batch_size)
    else:
        w_dash = np.random.beta(beta, beta, size=batch_size)
        h_dash = np.random.beta(beta, beta, size=batch_size)
        d_dash = np.random.beta(beta, beta, size=batch_size)
    w = np.round(w_dash * image_x).astype(np.int32)
    h = np.round(h_dash * image_y).astype(np.int32)
    d = np.round(d_dash * image_z).astype(np.int32)

    # outputs
    output_images = np.zeros(image_batch.shape)
    output_labels = np.zeros(label_batch.shape)

    def create_masks(start_xs, start_ys, start_zs, end_xs, end_ys, end_zs):
        mask_x = np.logical_and(np.arange(image_x).reshape(1, 1, 1, -1, 1) >= start_xs.reshape(-1, 1, 1, 1, 1),
                                np.arange(image_x).reshape(1, 1, 1, -1, 1) < end_xs.reshape(-1, 1, 1, 1, 1))
        mask_y = np.logical_and(np.arange(image_y).reshape(1, 1, -1, 1, 1) >= start_ys.reshape(-1, 1, 1, 1, 1),
                                np.arange(image_y).reshape(1, 1, -1, 1, 1) < end_ys.reshape(-1, 1, 1, 1, 1))
        mask_z = np.logical_and(np.arange(image_z).reshape(1, -1, 1, 1, 1) >= start_zs.reshape(-1, 1, 1, 1, 1),
                                np.arange(image_z).reshape(1, -1, 1, 1, 1) < end_zs.reshape(-1, 1, 1, 1, 1))

        mask_xy = np.logical_and(mask_y, mask_x)
        mask = np.logical_and(mask_xy, mask_z)
        mask = np.logical_and(mask, np.repeat(True, image_batch.shape[4]).reshape(1, 1, 1, 1, -1))
        return mask

    def crop_concatenate(wk, hk, dk, start_x, start_y, start_z, end_x, end_y, end_z):
        nonlocal output_images, output_labels

        xk = (np.random.rand(batch_size) * (image_x - wk)).astype(np.int32)
        yk = (np.random.rand(batch_size) * (image_y - hk)).astype(np.int32)
        zk = (np.random.rand(batch_size) * (image_z - dk)).astype(np.int32)
        target_indices = np.arange(batch_size)
        np.random.shuffle(target_indices)

        dest_mask = create_masks(start_x, start_y, start_z, end_x, end_y, end_z)
        target_mask = create_masks(xk, yk, zk, xk + wk, yk + hk, zk + dk)

        output_images[dest_mask] = image_batch[target_indices][target_mask]
        output_labels[dest_mask] = label_batch[target_indices][target_mask]

    # left-top crop
    crop_concatenate(w, h, d,
                     np.repeat(0, batch_size), np.repeat(0, batch_size), np.repeat(0, batch_size),
                     w, h, d)
    # right-top crop
    crop_concatenate(image_x - w, h, d,
                     w, np.repeat(0, batch_size), np.repeat(0, batch_size),
                     np.repeat(image_x, batch_size), h, d)
    # left-bottom crop
    crop_concatenate(w, image_y - h, d,
                     np.repeat(0, batch_size), h, np.repeat(0, batch_size),
                     w, np.repeat(image_y, batch_size), d)
    # right-bottom crop
    crop_concatenate(image_x - w, image_y - h, d,
                     w, h, np.repeat(0, batch_size),
                     np.repeat(image_x, batch_size), np.repeat(image_y, batch_size), d)
    # left-top crop 2
    crop_concatenate(w, h, image_z -d,
                     np.repeat(0, batch_size), np.repeat(0, batch_size), d,
                     w, h, np.repeat(image_z, batch_size))
    # right-top crop 2
    crop_concatenate(image_x - w, h, image_z -d,
                     w, np.repeat(0, batch_size), d,
                     np.repeat(image_x, batch_size), h, np.repeat(image_z, batch_size))
    # left-bottom crop 2
    crop_concatenate(w, image_y - h, image_z -d,
                     np.repeat(0, batch_size), h, d,
                     w, np.repeat(image_y, batch_size), np.repeat(image_z, batch_size))
    # right-bottom crop 2
    crop_concatenate(image_x - w, image_y - h, image_z -d,
                     w, h, d,
                     np.repeat(image_x, batch_size), np.repeat(image_y, batch_size), np.repeat(image_z, batch_size))

    return output_images, output_labels


class MyGenerator(Sequence):

    def __init__(self, data_paths, data_classes, batch_size, ch=1, beta=0, alpha=0, flip=1, ws=[2000, 400, 600, 150]):
        self.ricap_beta = beta
        self.mixup_alpha = alpha
        self.flip = flip
        self.ws = ws
        self.data_paths = data_paths
        self.data_classes = data_classes
        self.length = len(data_paths)
        self.batch_size = batch_size

        self.depth, self.height, self.width, _ = self.get_input_shape()
        self.ch = ch
        self.num_batches_per_epoch = self.length // batch_size - 1

        self.on_epoch_end()

        N = random.randint(0, 10000)
        random.seed(N)
        random.shuffle(self.data_paths)
        random.seed(N)
        random.shuffle(self.data_classes)

        return

    def __getitem__(self, idx):

        batch_X, batch_Y = self.__load(idx)

        if self.ricap_beta > 0:
            batch_X, batch_Y = ricap(batch_X, batch_Y, self.ricap_beta)

        if self.ws[0]:
            if len(self.ws) == 2:
                batch_X = window_setting_and_normalize(batch_X, self.ws[0], self.ws[1])
            elif len(self.ws) == 4:
                batch_X1 = window_setting_and_normalize(batch_X, self.ws[0], self.ws[1])
                batch_X2 = window_setting_and_normalize(batch_X, self.ws[2], self.ws[3])
                batch_X = np.concatenate([batch_X1, batch_X2], axis=4)
            elif len(self.ws) == 6:
                batch_X1 = window_setting_and_normalize(batch_X, self.ws[0], self.ws[1])
                batch_X2 = window_setting_and_normalize(batch_X, self.ws[2], self.ws[3])
                batch_X3 = window_setting_and_normalize(batch_X, self.ws[4], self.ws[5])
                batch_X = np.concatenate([batch_X1, batch_X2, batch_X3], axis=4)

        if self.mixup_alpha > 0:
            batch_X, batch_Y = mix_up(batch_X[:self.batch_size], batch_Y[:self.batch_size],
                                      batch_X[self.batch_size:], batch_Y[self.batch_size:], self.mixup_alpha)

        return batch_X, batch_Y

    def __load(self, idx):

        start_pos = self.batch_size * idx
        if self.mixup_alpha > 0:
            end_pos = start_pos + self.batch_size * 2
        else:
            end_pos = start_pos + self.batch_size

        if end_pos > self.length:
            end_pos = self.length
        item_paths = self.data_paths[start_pos: end_pos]
        item_classes = self.data_classes[start_pos: end_pos]

        imgs = np.empty((len(item_paths), self.height, self.width, self.depth, self.ch), dtype=np.float32)
        labels = np.empty((len(item_paths), self.height, self.width, self.depth, self.ch), dtype=np.float32)

        for i, (item_path, item_class) in enumerate(zip(item_paths, item_classes)):
            img = np.load(item_path)
            label = np.load(item_class)
            if self.flip == 1:
                if np.random.rand() > 0.5:
                    img = img[:, :, ::-1]
                    label = label[:, :, ::-1]
            img.shape = [self.height, self.width, self.depth, self.ch]
            label.shape = [self.height, self.width, self.depth, self.ch]
            imgs[i] = img
            labels[i] = label

        return imgs, labels

    def __len__(self):
        return self.num_batches_per_epoch

    def get_input_shape(self):

        arr_sample = np.load(self.data_paths[0])

        return arr_sample.shape[0], arr_sample.shape[1], arr_sample.shape[2], len(self.ws)//2

    def on_epoch_end(self):

        return
