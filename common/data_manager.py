import os, sys
import numpy as np
import pandas as pd

from log_manager import *

def read_pickle(pkl_path):
    pkl = pd.read_pickle(pkl_path)
    return pkl

def get_mean_std(data):
    return np.mean(data, axis=0), np.std(data, axis=0)

def reshape_btnf(x, y, frames, nodes, features, pred_itv):
    x = x.reshape(-1, nodes, features)
    num_input = x.shape[0] - frames - pred_itv
    _input = np.zeros(shape=[num_input, frames, nodes, features], dtype=float)

    for i in range(0, num_input):
        _input[i,:,:,:] = x[i:i+frames,:,:]

    _output = y[-num_input:,:]
    return _input, _output

def reshape_bhwc(x, y, h=1, w=12, c=22, pred_itv=3):
    num_input = x.shape[0] - w - pred_itv
    _input = np.zeros(shape=[num_input, h, w, c], dtype=float)

    for i in range(0, num_input):
        _input[i,:,:,:] = x[i:i+w,:]

    _output = y[-num_input:]
    return _input, _output

class Scaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class Data_Feeder:
    def __init__(self, x, y, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.size = x.shape[0]
        self.num_batch = int(self.size//self.batch_size)

        if shuffle == True:
            permutation = np.random.permutation(self.size)
            self.x, self.y = x[permutation], y[permutation]
        else:
            self.x, self.y = x, y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind   = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.x[start_ind:end_ind, ...]
                y_i = self.y[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class Data_Manager:
    def __init__(self, data_dir, log_manager, batch_size, reshape_type, frames, pred_itv, shuffle=False):
        self.data_dir   = data_dir
        self.lm         = log_manager
        self.batch_size = batch_size
        self._files     = ['train_x', 'train_y', 'valid_x', 'valid_y', 'test_x', 'test_y']
        self.data       = {}

        self.read()
        self.x_keys = self.data['train_x'].keys()
        self.y_keys = self.data['train_y'].keys()
        self.normalize()
        if reshape_type == 'btnf':
            self.reshape_btnf(frames, pred_itv)
        elif reshape_type == 'bhwc':
            self.reshape_bhwc(frames, pred_itv)
        self.gen_feeder(shuffle)

    def read(self):
        self.lm.add_line('Read data from: {}'.format(self.data_dir))
        for _file in self._files:
            pkl = _file + '.pkl'
            pkl_path = os.path.join(self.data_dir, pkl)
            self.data[_file] = read_pickle(pkl_path)
            self.lm.add_line('file: {}, shape: {}'.format(pkl, self.data[_file].shape))

    def normalize(self):
        self.lm.add_line('Normalization: z transform')
        mean_x, std_x = get_mean_std(self.data['train_x'])
        mean_y, std_y = get_mean_std(self.data['train_y'])
        self.lm.add_line('x mean shape: {}, y mean shape: {}'.format(mean_x.shape, mean_y.shape))
        self.x_scaler = Scaler(mean_x, std_x)
        self.y_scaler = Scaler(mean_y, std_y)

        for _file in self._files:
            if 'x' in _file:
                self.data[_file] = self.x_scaler.transform(self.data[_file]).values
            elif 'y' in _file:
                self.data[_file] = self.y_scaler.transform(self.data[_file]).values
            else:
                raise ValueError

    def denormalize(self, y):
        y = pd.DataFrame(y, columns=self.y_keys)
        y = self.y_scaler.inverse_transform(y)
        return y

    def reshape_btnf(self, frames, pred_itv):
        nodes = self.data['train_x'].shape[1]
        features = 1
        self.lm.add_line('Data reshape into [b, {}, {}, {}]'.format(frames, nodes, features))
        categories = ['train', 'valid', 'test']
        for category in categories:
            x_ = category + '_x'
            y_ = category + '_y'
            self.data[x_], self.data[y_] = reshape_btnf(self.data[x_], self.data[y_], frames, nodes, features, pred_itv)
            self.lm.add_line('{}: {}, {}'.format(category, self.data[x_].shape, self.data[y_].shape))

    def reshape_bhwc(self, frames, pred_itv):
        h = 1
        w = frames
        c = self.data['train_x'].shape[1]
        self.lm.add_line('Data reshape into [b, {}, {}, {}]'.format(h, w, c))
        categories = ['train', 'valid', 'test']
        for category in categories:
            x_ = category + '_x'
            y_ = category + '_y'
            self.data[x_], self.data[y_] = reshape_bhwc(self.data[x_], self.data[y_], h, w, c, pred_itv)
            self.lm.add_line('{}: {}, {}'.format(category, self.data[x_].shape, self.data[y_].shape))

    def gen_feeder(self, shuffle):
        self.data['tr_feeder'] = Data_Feeder(self.data['train_x'], self.data['train_y'], self.batch_size, shuffle=shuffle)
        self.data['va_feeder'] = Data_Feeder(self.data['valid_x'], self.data['valid_y'], self.batch_size, shuffle=False)
        self.data['te_feeder'] = Data_Feeder(self.data['test_x'],  self.data['test_y'],  self.batch_size, shuffle=False)

        self.lm.add_line('batch size: {}'.format(self.batch_size))
        self.lm.add_line('total number of batch: {}'.format(self.data['tr_feeder'].num_batch))

    def get_dims(self):
        return self.data['train_x'].shape[1], self.data['train_y'].shape[1]
