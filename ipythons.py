# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import torch
import torch.utils.data as data
from torch import cuda, optim, nn
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from urllib import error, request
import shutil
import glob
import pickle
import argparse
import time
import os

# %%


class Cifar(data.Dataset):
    def __init__(self, train=True, trans=None, path='./cifar'):
        self.path = path
        self.transforms = trans
        self, x, self.y = self.pre(train)

    def unpickle(self, filename):
        fo = open(filename, 'rb')
        return pickle.load(fo, encoding='latin-1')

    def pre(self, train):
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
        x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1,)
        return x, y
