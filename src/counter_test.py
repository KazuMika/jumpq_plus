# -*- coding: utf-8 -*-
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.experimental import attempt_load
from pathlib import Path
from tracker.iou_tracking import Iou_Tracker
from tracker.sort import Sort
import glob
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import cv2
from collections import deque
import random
import threading
import os
import csv
import unittest


class TestCounter(unittest.TestCase):
    def test_model(self):
        pass
