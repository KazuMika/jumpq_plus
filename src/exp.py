# -*- coding: utf-8 -*-
from utils.general import check_imshow, increment_path
from utils.dataloaders import LoadImages, LoadStreams
from pathlib import Path
import glob
import torch.backends.cudnn as cudnn
import torch
from collections import deque
import os
import csv
import time
from counter import Counter


cudnn.benchmark = True

VIDEO_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg',
                 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class Exp(object):
    def __init__(self, opt):
        """
        Initialize
        """
        self.opt = opt  # config
        self.source, self.view_img, self.save_txt, self.imgsz, self.save_movie = \
            opt.source, opt.view_img, opt.save_txt, opt.img_size, opt.save_movie
        self.save_dir = Path(increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        self.save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        (self.save_dir / 'detected_images').mkdir(parents=True, exist_ok=True)  # make dir
        (self.save_dir / 'detected_movies').mkdir(parents=True, exist_ok=True)  # make dir
        self.save_images_dir = self.save_dir / 'detected_images'
        self.save_movies_dir = self.save_dir / 'detected_movies'
        self.counting_mode = opt.counting_mode
        self.frame_num = 0
        self.save_image = opt.save_image
        self.number_exp = 0
        self.counter = Counter(self.opt)
        if opt.mode == 'webcam':
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
                    self.counter.counting(movie)
                else:
                    for movie_path in self.movies:
                        self.csv_writer = csv.writer(f)
                        self.csv_writer_fps = csv.writer(f2)
                        self.dataset = LoadImages(
                            movie_path, img_size=self.imgsz, stride=self.stride)
                        self.image_save_stack = deque()
                        print(movie_path)
                        self.counter.counting(movie_path)
                        if self.counting_mode == 'v1':
                            self.csv_writer.writerow([self.counter.cnt_down])
                        else:
                            while self.is_movie_opened or self.queue_images:
                                time.sleep(2.0)

                        self.csv_writer.writerow([self.counter.cnt_down])
                        self.csv_writer_fps.writerow([self.counter.frame_rate])
                        print(self.counter.cnt_down)

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
                                self.counter.counting(movie_path)
                                TP = 26
                                while self.is_movie_opened or self.queue_images:
                                    time.sleep(1.0)
                                self.ho.writerow(["{0:0.2f}".format(self.Ps), '{0:0.2f}'.format(self.LC),
                                                  self.Tw, self.K, '{:0.2f}'.format(self.counter.frame_rate), '{0:0.3f}'.format(self.counter.cnt_down / TP)])
                                self.csv_writer.writerow([self.counter.cnt_down])
                                print('Ps,Tw,LC,K', self.Ps,
                                      self.Tw, self.LC, self.K)
