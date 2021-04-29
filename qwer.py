# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import glob
import torch.utils.data as data
from torchvision import transforms
import pickle
import argparse
import time
import os
from torch import cuda, optim
from torch.utils import data
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
import tarfile
from urllib import request, error

ckpt = './checkpoint/adf.t7'
torch.backends.cudnn.benchmark = True


class Cifar(data.Dataset):
    def __init__(self, train=True, trans=None, path='./cifar'):
        self.path = path
        self.transforms = trans
        self.x, self.y = self.pre(train)

    def unpikcle(self, filename):
        fo = open(filename, 'rb')
        return pickle.load(fo, encodig='latin-1')

    def download(self):
        url = 'https://'
        extracted_dir = 'python_extracted_dir'
        save_filename = url.split('/')[-1]
        save_dir = './cifar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            try:
                with request.urlopen(url) as web_file, open(save_filename, 'wb') as local_file:
                    data = web_file.read()
                    local_file.write(data)

                with tarfile.open(save_filename) as tar:
                    tar.extractall()
                for cifar_file in glob.glob(os.path.join(extracted_dir, '*')):
                    shutil.move(cifar_file, save_dir)
                shutil.rmtree(extracted_dir)
                os.remove(save_filename)
            except error.URLError as e:
                print(e)
            return True

    def pre(self, train):
        self.download()
        if train:
            for i in range(1, 6):
                filename = os.path.join(self.path, 'data_batch_{}'.format(i))
                data_dict = self.unpickle(filename)
                if i == 1:
                    x, y = data_dict['data'], data_dict['labels']
                else:
                    x = np.vstack((x, data_dict['data']))
                    y = np.hstack((y, data_dict['labels']))
        else:
            filename = os.path.join(self.path, 'test_batch')
            data_dict = self.unpickle(filename)
            x, y = data_dict['data'], data_dict['labels']
        x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return x, y

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transforms is not None:
            x = self.transforms(Image.fromarray(x))
        return x, y

    def __len__(self):
        return len(self.x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp, oup, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inp, oup, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.conv2 = nn.Conv2d(oup, oup, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        slef.con
