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

from tracker.sort import Sort
from tracker.iou_tracking import Iou_Tracker
from utils.fpsrate import FpsWithTick
from pathlib import Path
from utils.count_utils import find_all_files
from detector.yolov5.setup_model import detect, make_model

cudnn.benchmark = True


class Counter(object):
    def __init__(self, args):
        self.time = time.time()
        self.fpsWithTick = FpsWithTick()
        self.frame_count = 0
        self.fps_count = 0
        self.p = 0
        self.q = deque()
        self.recallq = deque()
        self.flag_of_realtime = True
        self.args = args
        self.path = self.args.save_dir_path
        self.conf_thres = self.args.conf_thres
        self.nms_thres = self.args.nms_thres
        self.img_size = self.args.img_size
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.y_pred_list = []
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.max_age = 2
        self.frame_rate = 20
        self.mode = self.args.mode
        self.tracking_alg = self.args.tracking_alg
        self.model = args.model
        self.fps_eval = self.args.fps_eval
        self.video = self.args.video
        self.l = 10  # the lower bound of processing rate  by koyo
        self.w = 0  # the window size of frames in which objects are not detected
        self.save_root_dir = self.path

        if self.model == 'yolo':
            self.model, self.names, self.webcam = make_model(self.args)
        self.prepare()

        f = open(os.path.join(self.save_root_dir, 'y_pred.csv'), "w")
        self.writer = csv.writer(f, lineterminator='\n')

    def execution(self):
        if self.mode == 'precision':
            for movie in self.movies:
                print(movie)
                self.evalate_precision(movie)

            self.writer.writerows(self.y_pred_list)

        elif self.mode == 'visualization':
            for movie in self.movies:
                print(movie)
                datevs = movie.split('/')[5]

                if len(self.gps_datelist) == 0:
                    self.gps_datelist.append(datevs)
                    gps_log = open(os.path.join(
                        self.save_root_dir, self.gps_dir, self.gps_location_dir, datevs+'.csv'), 'w')
                    writer2 = csv.writer(gps_log, lineterminator='\n')
                    self.gps_list = []
                elif datevs not in self.gps_datelist:
                    self.gps_datelist.append(datevs)
                    writer2.writerows(self.gps_list)
                    gps_log.close()
                    gps_log = open(os.path.join(
                        self.save_root_dir, self.gps_dir, self.gps_location_dir, datevs+'.csv'), 'w')
                    self.gps_list = []
                    writer2 = csv.writer(gps_log, lineterminator='\n')

                self.visualization(movie)

            writer2.writerows(self.gps_list)
            gps_log.close()
            self.writer.writerows(self.y_pred_list)

        elif self.mode == 'jetson':
            self.counting_on_jetson()

        elif self.mode == 'realtime':
            for movie in self.movies:
                print(movie)
                self.realtime_detection(movie)

            self.writer.writerows(self.y_pred_list)
        else:
            pass

    def evalate_precision(self, path_to_movie):
        cap = cv2.VideoCapture(path_to_movie)
        basename = os.path.basename(path_to_movie).replace('.mp4', '')
        movie_id = basename[0:4]

        save_movie_path = os.path.join(self.movie_dir, basename+'.mp4')
        if self.video:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video = cv2.VideoWriter(save_movie_path, fourcc,
                                    self.frame_rate, (int(cap.get(3)), int(cap.get(4))))
        height = cap.get(4)
        line_down = int(9*(height/10))

        if self.tracking_alg == 'sort':
            tracker = Sort(1, self.max_age, line_down, movie_id,
                           self.image_dir, '', basename)
        else:
            tracker = Iou_Tracker(
                line_down, self.image_dir, movie_id, self.max_age, '', basename)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break

            if self.tracking_alg == 'ssd':
                cords = np.array(self.detect_image(rgb_image))
            else:
                cords = self.detect_image(rgb_image)

            tracker.update(cords, frame, fps_eval=self.fps_eval)

            if self.video:
                video.write(frame)

        if self.fps_eval:
            print("Avarage fps : {0:.2f}".format(
                tracker.fps_count / tracker.frame_count))

        self.y_pred_list.append((movie_id, tracker.cnt_down))
        cap.release()
        cv2.destroyAllWindows()

        if self.video:
            video.release()

        self.time = time.time()-self.time
        print('end_time:{}'.format(self.time))

    def visualization(self, path_to_movie):
        cap = cv2.VideoCapture(path_to_movie)
        basename = os.path.basename(path_to_movie)
        movie_date = path_to_movie.split('/')[-3]
        save_image_dir = os.path.join(
            self.save_root_dir, self.gps_dir, self.gps_image_dir)
        movie_id = basename[0:4]
        height = cap.get(4)
        line_down = int(9*(height/10))
        gps_path = path_to_movie.replace('H264/CH1.264', 'SNS/Sns.txt')
        if not os.path.exists(gps_path):
            return None
        f2 = open(gps_path, 'r')
        gpss = [gps.strip() for gps in f2.readlines()]
        gps_count = 0
        if self.tracking_alg == 'sort':
            tracker = Sort(1, 3, line_down, movie_id,
                           save_image_dir, movie_date)
        else:
            tracker = Iou_Tracker(line_down, save_image_dir,
                                  movie_id, 2, movie_date)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break

            if self.tracking_alg == 'sort':
                cords = np.array(self.detect_image(frame))
            else:

                cords = self.detect_image(rgb_image)
            tracker.update(cords, frame, gpss=gpss, gps_count=gps_count,
                           visualize=True, gps_list=self.gps_list)
            gps_count += 1

        cap.release()
        # video.release()
        cv2.destroyAllWindows()
        self.y_pred_list.append((movie_id, tracker.cnt_down))

    def counting_on_jetson(self):
        cap = cv2.VideoCapture('/home/quantan/DTCEvaluation/yolov3_dtceval/testgomi2.mp4')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # cap = cv2.VideoCapture('testgomi2.mp4')
        time_stamp = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        save_movie_dir = os.path.join(self.save_movie_dir, (time_stamp+'.avi'))
        if not os.path.exists(self.save_image_dir):
            os.mkdir(self.save_image_dir)

        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(save_movie_dir, fourcc,
                                20, (int(cap.get(3)), int(cap.get(4))))

        prediction = []
        prediction2 = []

        height = cap.get(4)
        line_down = int(9*(height/10))

        frame_count = 0
        fps_count = 0
        fpsWithTick = FpsWithTick()
        count = 0
        if self.tracking_alg == 'sort':
            tracker = Sort(1, 3, line_down,
                           save_image_dir=save_movie_dir)
        else:
            tracker = Iou_Tracker(line_down, save_image_dir=self.save_image_dir,
                                  save_movie_dir=save_movie_dir)

        while(cap.isOpened()):
            time_stamp = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
            count += 1
            ret, frame = cap.read()
            if ret:
                frame2 = frame.copy()
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break

            if self.tracking_alg == 'sort':
                cords = np.array(self.detect_image(rgb_image))
            else:
                cords = self.detect_image(rgb_image)
            tracker.update(cords, frame, prediction2=prediction2,
                           time_stamp=time_stamp, demo=True)

            video.write(frame2)

            fps1 = fpsWithTick.get()
            fps_count += fps1
            frame_count += 1
            if frame_count == 0:
                frame_count += 1

            if count % 50 == 0:
                f = open(os.path.join(self.save_root_dir, 'prediction.csv'), "a")
                f2 = open(os.path.join(self.save_root_dir, 'prediction2.csv'), "a")
                writer = csv.writer(f, lineterminator='\n')
                writer2 = csv.writer(f2, lineterminator='\n')
                time_stamp = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
                avg_fps = (fps_count / frame_count)
                fps_count = 0
                frame_count = 0
                prediction.append((time_stamp, tracker.cnt_down, avg_fps))
                writer.writerows(prediction)
                writer2.writerows(prediction2)
                save_movie_dir = os.path.join(self.save_movie_dir, (time_stamp+'.avi'))
                prediction = []
                prediction2 = []

                # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video = cv2.VideoWriter(save_movie_dir, fourcc,
                                        20, (int(cap.get(3)), int(cap.get(4))))
                f.close()

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        video.release()
        cv2.destroyAllWindows()

    def detect_image(self, frame):
        if self.model == 'yolo':
            return cords

    def realtime_detection(self, path_to_movie):
        cap = cv2.VideoCapture(path_to_movie)
        basename = os.path.basename(path_to_movie).replace('.mp4', '')
        movie_id = basename[0:4]

        save_movie_path = os.path.join(self.movie_dir, basename+'.mp4')
        print(save_movie_path)
        if self.video:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video = cv2.VideoWriter(save_movie_path, fourcc,
                                    self.frame_rate, (int(cap.get(3)), int(cap.get(4))))
        height = cap.get(4)
        line_down = int(9*(height/10))
        t1 = threading.Thread(target=self.recall_q2, args=(line_down, height, movie_id, basename))
        t1.start()
        i = 0

        while(cap.isOpened()):
            i += 1
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.q.append(rgb_image)
            else:
                self.flag_of_realtime = False
                break
            if self.video:
                video.write(frame)

    def recall_q2(self, line_down, height, movie_id, basename):
        LC = self.l/self.frame_rate
        Ps = 0.1
        Pd = 1
        Tw = 10
        # self.tracking_alg = 'iou'
        if self.tracking_alg == 'sort':
            tracker = Sort(self.max_age, 3, line_down, movie_id,
                           self.image_dir, '', basename)
        else:
            tracker = Iou_Tracker(line_down, self.image_dir, movie_id, self.max_age, '', basename)
        i = 0
        while self.flag_of_realtime or self.q:
            if self.q:
                i += 1
                newFrame = self.q.popleft()
                if newFrame is not None:
                    Ran = random.random()
                    if len(self.recallq) < 10:
                        self.recallq.append(newFrame)

                        continue
                    if Ran < Pd:
                        cords = self.detect_image(newFrame)
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
