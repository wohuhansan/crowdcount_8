# load data in multithreading

import numpy as np
import os
import cv2
import random
import pandas
import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import psutil
from os.path import join
import pickle
import string
import datetime

from src.utils import ndarray_to_tensor, print_red, make_path
from src.data_path import DataPath


DOWNSAMPLE = 8  # The output of the network needs to be downsampled to one of DOWNSAMPLE
NUMBER_OF_LABELS = 4


class PreloadData:
    def __init__(self, image_path, density_map_path, roi_path=None, is_preload=False, is_label=False, is_mask=False):
        # image_path: path of all image file
        # density_map_path: path of all density map file
        # roi_path: path of all region of interest file
        self.image_path = image_path
        self.density_map_path = density_map_path
        self.roi_path = roi_path
        self.is_preload = is_preload
        self.is_label = is_label
        self.is_mask = is_mask

        self.image_channel = 3
        self.min_available_memory = 8 * 1024 * 1024 * 1024  # 8GB

        # make path for pickle
        time_now = datetime.datetime.now()
        self.pickle_path = os.path.join(r'D:\PycharmProjects\pickle', '%4d%02d%02d%02d%02d%02d%06d_%s' %
                                        (time_now.year, time_now.month, time_now.day, time_now.hour, time_now.minute, time_now.second, time_now.microsecond, ''.join(random.sample(string.ascii_letters, 4))))
        make_path(self.pickle_path)

        # get all image file name
        self.image_filename_list = [filename for filename in os.listdir(self.image_path) if os.path.isfile(join(self.image_path, filename))]
        self.image_filename_list.sort()

        self.number_of_samples = len(self.image_filename_list)

        self.preload_data_dict = dict()  # store all preload data in this dict

        index = 0
        for filename in self.image_filename_list:
            if self.is_preload:
                if psutil.virtual_memory().available > self.min_available_memory:
                    index += 1
                    self.preload_data_dict[filename] = self.read_blob(filename)
                    if index % 100 == 0:
                        print('Loaded %6d of %d files.' % (index, self.number_of_samples))
                else:
                    self.preload_data_dict[filename] = None

            else:
                self.preload_data_dict[filename] = None
        print('Completed loading %d files. %d files are preloaded.' % (self.number_of_samples, index))

        return

    def get_number_of_samples(self):
        return self.number_of_samples

    def get_blob_by_index(self, index):
        filename = self.image_filename_list[index]
        this_blob = self.preload_data_dict[filename]

        if this_blob is None:  # no data is preloaded for this blob
            pickle_file_path = os.path.join(self.pickle_path, filename + '.pickle')
            if os.path.isfile(pickle_file_path):
                with open(pickle_file_path, 'rb') as file:
                    this_blob = pickle.load(file)
            else:
                this_blob = self.read_blob(filename)
                with open(pickle_file_path, 'wb') as file:
                    pickle.dump(this_blob, file)

        return this_blob

    def read_blob(self, filename):
        image_name, _ = os.path.splitext(filename)
        blob = dict()
        blob['image_name'] = image_name

        if self.image_channel == 1:
            image = cv2.imread(join(self.image_path, filename), 0)
        elif self.image_channel == 3:
            image = cv2.imread(join(self.image_path, filename), 1)
        else:
            raise Exception('invalid number of image channels')

        density_map = pandas.read_csv(join(self.density_map_path, image_name + '.csv'), sep=',', header=None).values

        if image.shape[0] != density_map.shape[0] or image.shape[1] != density_map.shape[1]:
            raise Exception('density map size mismatch.')

        density_map = self.downsample(density_map, DOWNSAMPLE)

        if self.roi_path is not None:
            if DOWNSAMPLE == 1:
                roi = self.load_roi(join(self.roi_path, image_name + '_roi.mat'))
            elif DOWNSAMPLE == 4:
                roi = self.load_roi(join(self.roi_path, image_name + '_roi_fourth_size.mat'))
            elif DOWNSAMPLE == 8:
                roi = self.load_roi(join(self.roi_path, image_name + '_roi_eighth_size.mat'))
            else:
                raise Exception('no suitable RoI file available')
        else:
            roi = None

        image = self.reshape_data(image)
        density_map = self.reshape_data(density_map)
        if roi is not None:
            roi = self.reshape_data(roi)
            if roi.shape[2] != density_map.shape[2] or roi.shape[3] != density_map.shape[3]:
                raise Exception('RoI size mismatch')

        blob['image'] = ndarray_to_tensor(image, is_cuda=False)
        blob['density'] = ndarray_to_tensor(density_map, is_cuda=False)
        if roi is not None:
            blob['roi'] = ndarray_to_tensor(roi, is_cuda=False)
        else:
            blob['roi'] = ndarray_to_tensor(np.ones_like(density_map), is_cuda=False)

        if self.is_label:
            blob['label'] = ndarray_to_tensor(self.get_label(blob['density']), is_cuda=False)

        if self.is_mask:
            blob['mask'] = ndarray_to_tensor(self.get_mask(blob['density']), is_cuda=False)

        return blob

    def compute_label(self, count):
        if count == 0:
            return 0
        # label = int(min(max(np.floor(np.log2(density * 3200 / DOWNSAMPLE ** 2)), 0), NUMBER_OF_LABELS - 1))
        label = int(min(max(np.floor(np.log2(count / 10)), 1), NUMBER_OF_LABELS - 1))
        return label

    def get_label(self, density_map):
        # density_map torch.Tensor shape=(1, 1, h, w)
        if density_map.shape[0] != 1:
            raise Exception('invalid density map shape')
        # average_density = torch.mean(density_map)
        count = torch.sum(density_map)
        label = np.zeros(NUMBER_OF_LABELS, dtype=np.int)
        label[self.compute_label(count)] = 1
        return label

    def get_mask(self, ground_truth_map):
        # ground_truth numpy.ndarray shape=(1, 1, h, w)
        raise Exception('mask is not supported yet')

    def get_label_weights(self):
        return self.label_weights

    @staticmethod
    def downsample(density_map, downsample_value=1):
        # height and width of output density map are about 1/[downsample_value] times that of original density map
        import torch
        import torch.nn.functional as functional

        if density_map.shape[0] % downsample_value != 0 or density_map.shape[1] % downsample_value != 0:
            raise Exception('density map size is not suitable for downsample value')

        density_map = density_map.reshape((1, 1, density_map.shape[0], density_map.shape[1]))
        if downsample_value > 1:
            density_map_tensor = torch.tensor(density_map, dtype=torch.float32)
            density_map_tensor = functional.avg_pool2d(density_map_tensor, downsample_value, stride=downsample_value)
            density_map = density_map_tensor.data.cpu().numpy()
            density_map = density_map * downsample_value * downsample_value
        density_map = density_map.reshape((density_map.shape[2], density_map.shape[3]))

        return density_map

    @staticmethod
    def load_roi(path):
        roi_mat = scio.loadmat(path)
        roi = roi_mat['roi']
        raw_mask = roi['mask']
        mask = raw_mask[0, 0]
        mask = mask.astype(np.float32, copy=False)
        return mask

    @staticmethod
    def reshape_data(data):
        # data numpy.ndarray shape=(x, y) or (x, y, 3)
        # return numpy.ndarray shape=(1, x, y) or (3, x, y)
        data = data.astype(np.float32, copy=False)
        height = data.shape[0]
        width = data.shape[1]
        if len(data.shape) == 3 and data.shape[2] == 3:
            # image_r = data[:, :, 0]
            # image_g = data[:, :, 1]
            # image_b = data[:, :, 2]
            # image = np.zeros((3, height, width), dtype=np.float32)
            # image[0] = image_r
            # image[1] = image_g
            # image[2] = image_b
            data = np.moveaxis(data, 2, 0)
            reshaped_data = data.reshape((3, height, width))
        elif len(data.shape) == 2:
            reshaped_data = data.reshape((1, height, width))
        else:
            raise Exception('Invalid data shape.')

        return reshaped_data


class Data(Dataset):
    def __init__(self, preload_data):
        # image_path: path of all image file
        # density_map_path: path of all density map file
        # roi_path: path of all region of interest file
        self.preload_data = preload_data

    def __len__(self):
        return self.preload_data.get_number_of_samples()

    def __getitem__(self, index):
        return self.preload_data.get_blob_by_index(index)


def multithread_dataloader(data_config):
    # data_config: dict, a dictionay contains several datasets info,
    #              key is dataset name,
    #              value is a dict which contains is_preload and is_label and is_mask
    data_path = DataPath()

    data_dict = dict()

    for name in data_config:
        this_dataset_flag = data_config[name]
        is_preload = this_dataset_flag['preload']
        if 'label' in this_dataset_flag:
            is_label = this_dataset_flag['label']
        else:
            is_label = False
        if 'mask' in this_dataset_flag:
            is_mask = this_dataset_flag['mask']
        else:
            is_mask = False
        if 'shuffle' in this_dataset_flag:
            is_shuffle = this_dataset_flag['shuffle']
        else:
            is_shuffle = False
        if 'seed' in this_dataset_flag:
            random_seed = this_dataset_flag['seed']
        else:
            random_seed = None
        if 'batch_size' in this_dataset_flag:
            batch_size = this_dataset_flag['batch_size']
        else:
            batch_size = 1

        if random_seed is not None:
            def worker_init_fn(x):
                seed = random_seed + x
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                return
        else:
            worker_init_fn = None

        path = data_path.get_path(name)
        preload_data = PreloadData(path['image'], path['gt'], roi_path=path['roi'], is_preload=is_preload, is_label=is_label, is_mask=is_mask)
        this_data = Data(preload_data)
        this_dataloader = DataLoader(this_data, batch_size=batch_size, shuffle=is_shuffle, num_workers=8, drop_last=False, worker_init_fn=worker_init_fn)

        if is_label:
            number_of_samples = preload_data.get_number_of_samples()
            label_histogram = np.zeros(NUMBER_OF_LABELS)
            index = 0
            for blob in this_dataloader:
                labels = torch.argmax(blob['label'], dim=1, keepdim=True)
                for this_label in labels:
                    label_histogram[this_label] += 1
                index += 1
                if index % 100 == 0:
                    print('Built %6d of %d labels.' % (index, number_of_samples))

            print('Completed building %d labels. Label histogram is %s' % (index, ' '.join([str(i) for i in label_histogram])))
            label_weights = 1 - label_histogram / sum(label_histogram)
            label_weights = label_weights / sum(label_weights)
        else:
            label_weights = None

        this_dataset_dict = dict()
        this_dataset_dict['data'] = this_dataloader
        if is_label:
            this_dataset_dict['label_weights'] = ndarray_to_tensor(label_weights, is_cuda=False)

        data_dict[name] = this_dataset_dict

    return data_dict
