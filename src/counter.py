# -*- coding: utf-8 --
import time
import os
import csv
import threading
import datetime
import random
from collections import deque
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import glob
import yolov5

from tracker.sort import Sort
from tracker.iou_tracking import Iou_Tracker
from utils.fpsrate import FpsWithTick
from pathlib import Path
from utils.count_utils import find_all_files
import argparse

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized

cudnn.benchmark = True

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class Counter(object):
    def __init__(self, opt):
        self.opt = opt
        self.source, weights, self.view_img, self.save_txt, self.imgsz, self.save_img = \
            opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_img
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        self.save_dir = save_dir
        self.mode = opt.mode

        # Initialize
        set_logging()
        device = select_device(opt.device)
        self.half = device.type != 'cpu'  # half precision only supported on CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.max_age = 3
        self.tracking_alg = 'iou'

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.vid_path, self.vid_writer = None, None
        self.dataset = None
        self.images_q = deque()
        self.detection_q = deque()

        if self.half:
            self.model.half()  # to FP16

        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(device).type_as(next(self.model.parameters())))  # run once
        if self.mode == 'webcam':
            self.movies = []
            self.webcam = True
        else:
            self.movies = self.get_movies(self.source)
            self.webcam = False

    def get_movies(self, path):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        return videos

    def excute(self):
        with torch.no_grad():
            if self.webcam:
                movie = '0'
                self.view_img = check_imshow()
                cudnn.benchmark = True
                self.dataset = LoadStreams(movie, img_size=self.imgsz, stride=self.stride)
                self.realtime_detection(movie)
            else:
                for movie in self.movies:
                    self.dataset = LoadImages(movie, img_size=self.imgsz, stride=self.stride)
                    self.count(movie)

#     def realtime_detection(self,  path_to_movie):
#         basename = os.path.basename(path_to_movie).replace('.mp4', '')
#         movie_id = basename[0:4]
#         save_movie_path = os.path.join(self.save_dir.name, basename+'.mp4')
#         print(save_movie_path)
#         height = self.dataset.height
#         line_down = int(9*(height/10))
#         t1 = threading.Thread(target=self.recall_q2, args=(line_down, height, movie_id, basename))
#         t1.start()
#         time_0 = time.time()
#         for path, img, im0s, vid_cap in self.dataset:
#             self.vid_cap = vid_cap
#
#             img = torch.from_numpy(img).to(self.device)
#             img = img.half() if self.half else img.float()  # uint8 to fp16/32
#             img /= 255.0  # 0 - 255 to 0.0 - 1.0
#             if img.ndimension() == 3:
#                 img = img.unsqueeze(0)
#
#             self.images_q.append([img, im0s, path])
#             self.detect(self.images_q.popleft())
#
#         self.flag_of_realtime = False
#         if self.save_txt or self.save_img:
#             s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
#             print(f"Results saved to {self.save_dir}{s}")
#
#         print(f'Done. ({time.time() - time_0:.3f}s)')
#

    def count(self,  path_to_movie):
        basename = os.path.basename(path_to_movie).replace('.mp4', '')
        movie_id = basename[0:4]
        save_movie_path = os.path.join(self.save_dir.name, basename+'.mp4')
        print(save_movie_path)
        self.image_dir = './'
        height = self.dataset.height
        line_down = int(9*(height/10))

        if self.tracking_alg == 'sort':
            tracker = Sort(max_age=self.max_age,
                           line_down=line_down,
                           movie_id=movie_id,
                           save_image_dir='./runs',
                           movie_date='',
                           basename=basename,
                           min_hits=3)
        else:
            tracker = Iou_Tracker(max_age=self.max_age, line_down=line_down,
                                  save_image_dir=self.image_dir,
                                  movie_id=movie_id,
                                  movie_date='',
                                  base_name=basename)
        for path, img, im0s, vid_cap in self.dataset:
            self.vid_cap = vid_cap

            img2 = img.copy()

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            self.images_q.append([img, im0s, path])
            result = self.detect(self.images_q.popleft())
            # print(result)
            tracker.update(result, img2)

        self.flag_of_realtime = False
        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")

    def recall_q2(self, line_down, height, movie_id, basename):
        LC = self.l/self.frame_rate
        Ps = 0.1
        Pd = 1
        Tw = 10
        # self.tracking_alg = 'iou'
        if self.tracking_alg == 'sort':
            tracker = Sort(self.max_age, line_down, movie_id,
                           self.image_dir, '', basename, min_hits=3)
        else:
            tracker = Iou_Tracker(
                line_down, self.image_dir, movie_id, self.max_age, '', basename)

        i = 0
        ########################
        while self.flag_of_realtime or self.q:
            if self.q:
                i += 1
                newFrame = self.images_q.popleft()
                if newFrame is not None:
                    Ran = random.random()
                    if len(self.recallq) < 10:
                        self.detection_q.append(newFrame)
                        continue

                    if Ran < Pd:
                        cords = self.detect(newFrame)
                        if cords:
                            Pd = 1
                            self.w = 0
                            while self.recallq:
                                img = self.recallq.popleft()
                                detectQ = self.detect_image(img)
                                tracker.update(detectQ, img)

                        else:
                            self.w += 1
                            if self.w >= Tw:
                                Pd = max(Pd - Ps, LC)
                    else:
                        if Tw > len(self.recallq):
                            self.recallq.append(newFrame)
                        else:
                            self.recallq.append(newFrame)
                            self.recallq.popleft()
                else:
                    continue
        self.time = time.time()-self.time
        print('end_time:{}'.format(self.time))

    def detect(self, images):
        img, im0s, path = images
        pred = self.model(img, augment=self.opt.augment)[0]

        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        result = []
        for i, det in enumerate(pred):  # detections per image
            if self.webcam:  # batch_size >= 1
                p, s, im0, _ = path[i], '%g: ' % i, im0s[i].copy(), self.dataset.count
            else:
                p, s, im0, _ = path, '', im0s, getattr(self.dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det = det.cpu().numpy()[0]
                det = det.astype(np.int64)
                cord = det[:4]
                result.append(np.array(cord))

            if self.save_img:
                if self.dataset.mode == 'image':
                    cv2.imwrite(self.save_path, im0)
                else:  # 'video' or 'stream'
                    if self.vid_path != save_path:  # new video
                        self.vid_path = save_path
                        if isinstance(self.vid_writer, cv2.VideoWriter):
                            self.vid_writer.release()  # release previous video writer
                        if self.vid_cap:  # video
                            fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        return np.array(result)
