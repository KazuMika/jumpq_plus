# -*- coding: utf-8 -*-
import os
import torch.backends.cudnn as cudnn
import argparse
from counter import Counter
import time

cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/exp2/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/home/quantan/back_kanen_1_8_data_mp4/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--tracking_alg', '-t', type=str, default='iou', help="iou or sort")

    parser.add_argument("--mode", default='video', help='webcam or video')
    parser.add_argument("--save_img", action='store_true', help='save image optim default is False')
    opt = parser.parse_args()
    print(opt)
    counter = Counter(opt)
    counter.excute()


if __name__ == "__main__":

    main()
