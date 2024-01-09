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
import time


cudnn.benchmark = True

VIDEO_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg',
                 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

LC = 0.25  # LC = 0.3
Ps = 0.25  # Ps = 0.2
Tw = 10  # Tw = 10
K = 16  # K = 10


class Counter(object):
    def __init__(self, opt):
        """
        Initialize
        """
        self.opt = opt  # config
        self.cnt_down = self.pre_cnt_down = 0
        self.line_down = 0
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.source, weights, self.view_img, self.save_txt, self.imgsz, self.save_movie = \
            opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_movie
        self.save_dir = Path(increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        self.save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        self.weights = weights
        (self.save_dir / 'detected_images').mkdir(parents=True, exist_ok=True)  # make dir
        (self.save_dir / 'detected_movies').mkdir(parents=True, exist_ok=True)  # make dir
        self.save_images_dir = self.save_dir / 'detected_images'
        self.save_movies_dir = self.save_dir / 'detected_movies'
        self.mode = opt.mode
        self.counting_mode = opt.counting_mode
        self.frame_num = 0
        self.save_image = opt.save_image
        self.number_exp = 0

        # for jumpQ
        self.is_movie_opened = True
        self.queue_images = deque()

        # set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.max_age = 2 if self.opt.tracking_alg == 'sort' else 3
        self.tracking_alg = opt.tracking_alg

        # Load model
        self.model = attempt_load(
            weights, device=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        self.vid_path, self.vid_writer = None, None
        self.dataset = None

        if self.half:
            self.model.half()  # to FP16

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(
                self.device).type_as(next(self.model.parameters())))  # run once
        if self.mode == 'webcam':
            self.movies = []
            self.webcam = True
        else:
            self.movies = self.get_movies(self.source)
            self.webcam = False

    def get_movies(self, path):
        """
        pathから検出する動画を取得する

        Parameters
        ----------
        path : str
            動画もしくはディレクトリまでの相対パス

        Returns
        -------
        videos : list
            DTCで計数する動画までの絶対パスが入っているlist
        """
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        videos = [x for x in files if x.split(
            '.')[-1].lower() in VIDEO_FORMATS]
        return videos

    def excute(self):
        """
        実際にcountingを実行する
        検出をする際はこのメソッドを使用する
        """
        with torch.no_grad():
            with open(self.save_dir / 'count_result.csv', 'w') as f, open(self.save_dir / 'fps_result.csv', 'w') as f2:
                if self.webcam:
                    movie = '0'
                    self.view_img = check_imshow()
                    cudnn.benchmark = True
                    self.dataset = LoadStreams(
                        movie, img_size=self.imgsz, stride=self.stride)
                    self.counting(movie)
                else:
                    for movie_path in self.movies:
                        self.csv_writer = csv.writer(f)
                        self.csv_writer_fps = csv.writer(f2)
                        self.dataset = LoadImages(
                            movie_path, img_size=self.imgsz, stride=self.stride)
                        self.image_save_stack = deque()
                        print(movie_path)
                        self.cnt_down = self.pre_cnt_down = 0
                        self.counting(movie_path)
                        if self.counting_mode == 'v1':
                            self.csv_writer.writerow([self.cnt_down])
                        else:
                            while self.is_movie_opened or self.queue_images:
                                time.sleep(2.0)

                        self.csv_writer.writerow([self.cnt_down])
                        self.csv_writer_fps.writerow([self.frame_rate])
                        print(self.cnt_down)

    def excute_grid_search(self):
        """
        実際にcountingを実行する
        検出をする際はこのメソッドを使用する
        """
        with open(self.save_dir / 'hyperparameter_optimization.csv', 'w') as hy, \
                open(self.save_dir / 'count_result.csv', 'w') as f:
            self.ho = csv.writer(hy)
            self.csv_writer = csv.writer(f)
            self.number_exp = 0
            K = 10
            with torch.no_grad():
                movie_path = self.hs.pop()
                for Tw in list(range(1, 11)) + [15, 20, 25]:
                    for Ps in [1] + list(range(5, 100, 5)):
                        for K in [3, 4, 6, 7, 8, 9]:
                            for LC in [1] + list(range(5, 100, 5)):
                                self.number_exp += 1
                                self.Ps = Ps * 0.01
                                self.K = K
                                self.Tw = Tw
                                self.LC = LC * 0.01
                                self.dataset = LoadImages(
                                    movie_path, img_size=self.imgsz, stride=self.stride)
                                self.cnt_down = self.pre_cnt_down = 0
                                self.counting(movie_path)
                                TP = 26
                                while self.is_movie_opened or self.queue_images:
                                    time.sleep(1.0)
                                self.ho.writerow(["{0:0.2f}".format(self.Ps), '{0:0.2f}'.format(self.LC),
                                                  self.Tw, self.K, '{:0.2f}'.format(self.frame_rate), '{0:0.3f}'.format(self.cnt_down/TP)])
                                self.csv_writer.writerow([self.cnt_down])
                                print('Ps,Tw,LC,K', self.Ps,
                                      self.Tw, self.LC, self.K)
                                print("Recall:{0:0.2f}".format(
                                    self.cnt_down/TP))
                                print(self.cnt_down)