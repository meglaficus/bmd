from keras.utils import Sequence
from scipy.ndimage.interpolation import rotate
import random
import numpy as np


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


def mix_up(x1, y1, x2, y2, alpha):

    assert x1.shape[0] == y1.shape[0] == x2.shape[0] == y2.shape[0]

    batch_size = x1.shape[0]
    l = np.random.beta(alpha, alpha, batch_size)
    x_l = l.reshape(batch_size, 1, 1, 1, 1)
    y_l = l.reshape(batch_size, 1)
    x = x1 * x_l + x2 * (1 - x_l)
    y = y1 * y_l + y2 * (1 - y_l)

    return x, y


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
        weights = wk * hk * dk / image_x / image_y / image_z

        dest_mask = create_masks(start_x, start_y, start_z, end_x, end_y, end_z)
        target_mask = create_masks(xk, yk, zk, xk + wk, yk + hk, zk + dk)

        output_images[dest_mask] = image_batch[target_indices][target_mask]
        output_labels += weights.reshape(-1, 1) * label_batch[target_indices]

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





def xshift(im, x):
    xx = np.random.randint(-x, x)
    if xx == 0:
        return im
    else:
        if xx > 0:
            im[:,:,xx:] = im[:,:,:-xx]
            im[:,:,:xx] = 0
        else:
            xx = abs(xx)
            im[:,:,:-xx] = im[:,:,xx:]
            im[:,:,-xx:] = 0
        return im

def yshift(im, y):
    yy = np.random.randint(-y, y)
    if yy == 0:
        return im
    else:
        if yy > 0:
            im[:,yy:,:] = im[:,:-yy,:]
            im[:,:yy,:] = 0
        else:
            yy = abs(yy)
            im[:,:-yy,:] = im[:,yy:,:]
            im[:,-yy:,:] = 0
        return im

def zshift(im, z):
    zz = np.random.randint(-z, z)
    if zz == 0:
        return im
    else:
        if zz > 0:
            im[zz:,:,:] = im[:-zz,:,:]
            im[:zz,:,:] = 0
        else:
            zz = abs(zz)
            im[:-zz,:,:] = im[zz:,:,:]
            im[-zz:,:,:] = 0
        return im


class MyGenerator(Sequence):

    def __init__(self, data_paths, data_classes, batch_size=32, width=32, ch=1, num_of_class=2, beta=0, alpha=0, idg=0, flip=0, rs=[0,0], rot=0, ws = [None, None]):
        self.ricap_beta = beta
        self.mixup_alpha = alpha
        self.idg = idg
        self.flip = flip
        self.rs = rs
        self.rot = rot
        self.ws = ws
        self.data_paths = data_paths
        self.data_classes = data_classes
        self.length = len(data_paths)
        self.batch_size = batch_size
        self.width = width
        self.ch = ch

        self.num_of_class = num_of_class
        self.num_batches_per_epoch = self.length // batch_size - 1  # discard last one patch

        self.on_epoch_end()

        n = random.randint(0, 10000)
        np.random.seed(n)
        np.random.shuffle(self.data_paths)
        np.random.seed(n)
        np.random.shuffle(self.data_classes)


    def __getitem__(self, idx):

        batch_X, batch_Y = self.__load(idx)

        if self.mixup_alpha > 0:
            batch_X, batch_Y = mix_up(batch_X[:self.batch_size], batch_Y[:self.batch_size], batch_X[self.batch_size:], batch_Y[self.batch_size:], self.mixup_alpha)

        if self.ricap_beta > 0:
            batch_X, batch_Y = ricap(batch_X, batch_Y, self.ricap_beta)

        return batch_X, batch_Y


    def __load(self, idx):

        start_pos = self.batch_size * idx
        if self.mixup_alpha > 0:
            end_pos = start_pos + self.batch_size * 2
        else:
            end_pos = start_pos + self.batch_size

        if end_pos > self.length:
            end_pos = self.length
        item_paths = self.data_paths[start_pos:end_pos]
        item_classes = self.data_classes[start_pos:end_pos]

        imgs = np.empty((len(item_paths), self.width, self.width, self.width, self.ch), dtype=np.float32)
        labels = np.empty((len(item_paths), self.num_of_class), dtype=np.float32)


        for i, (item_path, item_class) in enumerate(zip(item_paths, item_classes)):
            img = np.load(item_path)

            if self.flip == 1:
                if np.random.rand() > 0.5:
                    img = img[:, :, ::-1]
            if self.rs[0] > 0:
                if np.random.rand() < self.rs[1]:
                    img = xshift(img, self.rs[0])
                    img = yshift(img, self.rs[0])
                    img = zshift(img, self.rs[0])
            if self.rot == 1:
                aaa = np.random.randint(-15, 15)
                img = rotate(img, angle=aaa, axes=(1, 2), reshape=False, cval=-1000)

            label = item_class
            img.shape = [self.width, self.width, self.width, self.ch]

            imgs[i] = img
            labels[i] = label

        if self.ws[0]:
            if len(self.ws) == 2:
                imgs = window_setting_and_normalize(imgs, self.ws[0], self.ws[1])
            elif len(self.ws) == 4:
                imgs1 = window_setting_and_normalize(imgs, self.ws[0], self.ws[1])
                imgs2 = window_setting_and_normalize(imgs, self.ws[2], self.ws[3])
                imgs = np.concatenate([imgs1, imgs2], axis=4)
            elif len(self.ws) == 6:
                imgs1 = window_setting_and_normalize(imgs, self.ws[0], self.ws[1])
                imgs2 = window_setting_and_normalize(imgs, self.ws[2], self.ws[3])
                imgs3 = window_setting_and_normalize(imgs, self.ws[4], self.ws[5])
                imgs = np.concatenate([imgs1, imgs2, imgs3], axis=4)

        else:
            imgs = imgs / 1000

        return imgs, labels

    def __len__(self):

        return self.num_batches_per_epoch

    def get_input_shape(self):

        arr_sample = np.load(self.data_paths[0])

        return arr_sample.shape[0], arr_sample.shape[1], arr_sample.shape[2], len(self.ws)//2

    def on_epoch_end(self):

        return



class BonePatchSequence(Sequence):

    def __init__(self, a, L=32, B=1, ws=[None, None], with_coordinate=False):

        self.l = L
        self.r = L // 2

        self.ct = a.ct
        self.bs = a.bs
        self.batch_size = B
        self.ws = ws

        self.lb = a.lb_0or1

        self.width = self.ct.shape[1]
        self.height = self.ct.shape[0]
        self.margin_xy = (self.width % self.r) // 2
        self.margin_z = (self.height % self.r) // 2

        self.array_xy = np.arange(self.margin_xy, self.width - self.l, self.r)
        self.array_z = np.arange(self.margin_z, self.height - self.l, self.r)

        self.nn = len(self.array_xy)
        self.nnn = len(self.array_z)

        self.coordinates_list = self.get_coordinates_list()
        self.with_coordinate = with_coordinate

    def get_coordinates_list(self):

        c = []
        for i in range(self.nnn):
            for j in range(self.nn ** 2):
                x = self.array_xy[j // self.nn]
                y = self.array_xy[j % self.nn]
                z = self.array_z[i]

                if self.l == 96:
                    if self.bs[z + 24:z + 72, y + 24:y + 72, x + 24:x + 72, 0].sum() > 1000:
                        c.append((x, y, z))
                else:
                    if self.bs[z:z + self.l, y:y + self.l, x:x + self.l, 0].sum() > 1000:
                        c.append((x, y, z))
        return c

    def get_gt_list(self):

        gt_list = [0] * len(self.coordinates_list)
        for i, c in enumerate(self.coordinates_list):
            x, y, z = c
            if self.lb[z:z + self.l, y:y + self.l, x:x + self.l, 0].sum() >= 1000:
                gt_list[i] = 1
            if 1000 > self.lb[z:z + self.l, y:y + self.l, x:x + self.l, 0].sum() > 0:
                gt_list[i] = 2
        return gt_list


    def __getitem__(self, idx):

        st = self.batch_size * idx
        en = st + self.batch_size
        if en > len(self.coordinates_list):
            en = len(self.coordinates_list)

        coordinates = self.coordinates_list[st:en]

        imgs = np.empty((en - st, self.l, self.l, self.l, 1), dtype=np.float32)

        for i, c in enumerate(coordinates):
            x, y, z = c
            im = self.ct[z:z + self.l, y:y + self.l, x:x + self.l]
            imgs[i, :, :, :, :] = im

        if self.ws[0]:
            if len(self.ws) == 2:
                imgs = window_setting_and_normalize(imgs, self.ws[0], self.ws[1])
            elif len(self.ws) == 4:
                imgs1 = window_setting_and_normalize(imgs, self.ws[0], self.ws[1])
                imgs2 = window_setting_and_normalize(imgs, self.ws[2], self.ws[3])
                imgs = np.concatenate([imgs1, imgs2], axis=4)
            elif len(self.ws) == 6:
                imgs1 = window_setting_and_normalize(imgs, self.ws[0], self.ws[1])
                imgs2 = window_setting_and_normalize(imgs, self.ws[2], self.ws[3])
                imgs3 = window_setting_and_normalize(imgs, self.ws[4], self.ws[5])
                imgs = np.concatenate([imgs1, imgs2, imgs3], axis=4)
        else:
            imgs = imgs / 1000

        if self.with_coordinate:
            return imgs, coordinates[0]
        else:
            return imgs


    def __len__(self):

        N = len(self.coordinates_list)
        B = self.batch_size

        if N % B == 0:
            return int(N / B)
        else:
            return int(N / B) + 1


    def on_epoch_end(self):

        pass




