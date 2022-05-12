import keras.utils.data_utils
import tensorflow as tf
import numpy as np
from keras.utils.data_utils import Sequence
import math


class DataGen(Sequence):
    def __init__(self, x, yt, batch_size, val_split=0.2):
        self.x, self.yt = x, yt
        self.batch_size = batch_size
        self.val_split = val_split
        # self.indices = np.arange(self.batch_size)
        self.ds_container = []
        self.ratio = 0
        self.xyt_c = []
        self.xyt_t = []
        self.create_dataset()

    def __len__(self):
        if len(self.x) % self.batch_size < 10:
            return math.floor(len(self.x) / self.batch_size)
        else:
            return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # batch = np.array(self.ds_container[idx*self.batch_size:(idx+1)*self.batch_size, :])
        n_treatment_samples = int(np.floor((1-self.ratio) * self.batch_size))
        n_control_samples = int(np.ceil(self.ratio * self.batch_size))
        idx_t = np.random.randint(len(self.xyt_t), size=n_treatment_samples)
        idx_c = np.random.randint(len(self.xyt_c), size=n_control_samples)
        batch = np.concatenate([self.xyt_c[idx_c], self.xyt_t[idx_t]], 0)
        np.random.shuffle(batch)
        batch_x = batch[:, :self.x.shape[1]]
        batch_yt = batch[:, self.x.shape[1]:]

        return batch_x, batch_yt

    def create_dataset(self):
        t = self.yt[:, 1:2].numpy()
        xyt = np.concatenate([self.x, self.yt], 1)
        c_idx = np.where(t == 0)[0].tolist()
        t_idx = np.where(t == 1)[0].tolist()
        self.ratio = float(len(c_idx) / (len(c_idx) + len(t_idx)))

        tf.random.set_seed(0)
        self.xyt_c = xyt[c_idx]
        self.xyt_t = xyt[t_idx]



