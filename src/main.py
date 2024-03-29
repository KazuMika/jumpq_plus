# -*- coding: utf-8 -*-
import torch.backends.cudnn as cudnn
import argparse
from exp import Exp

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', type=str, default=None, help='task')
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./sample_movie.mp4', help='source')  # file/folder, 0 for webcam
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
    parser.add_argument('--project', default='results', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--tracking_alg', '-t', type=str, default='iou', help="iou or sort")

    parser.add_argument("--mode", default='video', help='webcam or video')
    parser.add_argument("--counting_mode", default='v1', help='v1 or v2')
    parser.add_argument("--save_movie", action='store_true', default=False, help='save movie optim default is False')
    parser.add_argument("--save_image", action='store_true', default=False, help='save image optim default is False')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()
    if opt.which == "predict":
        exp = Exp(opt)
        exp.excute()
        # exp.excute_grid_search()


if __name__ == "__main__":

    main()
