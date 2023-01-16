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


python main.py --tracking_alg iou --counting_mode v1 --mode video --source /home/quantan/DTCdataset/DTC_counting_test_movie_1_8

# v1 & jetson nano & sort
#python main.py --counting_mode v2 --mode video --tracking_alg sort 




# if [ $# -ne 1 ]; then
#   echo "実行するには1個の引数が必要です。" 1>&2
#   exit 1
# fi
# 
# 
# if [ $TEST_VAR -eq 0 -o $TEST_VAR -eq 1 ]; then
#     echo "TEST_VARの値は0もしくは1です"
# elif [ $TEST_VAR -eq 2 ]; then
#     echo "TEST_VARの値は2です"
# elif [ $TEST_VAR -eq 3 ]; then
#     : #何もしない
# else
#     #if文のなかでif文が記述できる。
#     if [ $TEST_VAR_NEST_EX -eq 1 ]; then
# 	echo "TEST_VAR_NEST_EXは1です"
#     fi
# fi
