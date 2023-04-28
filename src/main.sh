#!/bin/sh


# v1 & video
#python main.py

# v1 & jetson nano
#python main.py --mode webcam

# v2 & video
#python main.py --counting_mode vide

# v2 & jetson nano
#python main.py --counting_mode v2 --mode webcam

# v2 & video & sort
#python main.py --tracking_alg iou --counting_mode v1 --mode video


python main.py --tracking_alg iou --counting_mode v2 --mode video --source ~/DTCdataset/DTC_counting_test_movie_1_8
#python main.py --tracking_alg iou --counting_mode v2 --mode video --source ./sample_movie.mp4 --device 1

# python main.py --tracking_alg iou --counting_mode v2 --mode video --source ./sample_movie.mp5
# v1 & jetson nano & sort
#python main.py --counting_mode v2 --mode video --tracking_alg sort 



