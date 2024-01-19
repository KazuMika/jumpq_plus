#!/bin/bash

DATE=$(date +%Y.%m.%d_%H.%M.%S)
HERE=$(pwd)
LOG_DIR="${HERE}"/.log
if [ ! -d "${LOG_DIR}" ]; then 
    mkdir "${LOG_DIR}"; 
fi
which=${1:-run}
THIS_START=$(date +%Y.%m.%d_%H.%M.%S)
echo "[START: $which] [$THIS_START] ######################################################################"


export PYTHONPATH=${PYTHONPATH}:${HERE}/src/yolov5
export PYTHONPATH=${PYTHONPATH}:${HERE}/src/

if [ $# -gt 1 ]; then
    script=$0
    for which in "$@"; do
        "${script} ${which}"
    done
    exit
fi

if [ "${which}" == "train" ]; then
    weights="yolov5s.pt"
    # project="${HERE}/results/runs/train"
    data="dtc.yaml"
    batch_size=8
    epochs=100
    resume="--resume"

    python ./src/yolov5/train.py \
        --weights "${weights}" \
        --data "${data}" \
        --batch-size ${batch_size} \
        --epochs ${epochs} 2>&1 |  tee "$LOG_DIR/$which.log.$DATE"
fi

if [ "${which}" == "predict" ]; then
    src="${HERE}"
    tracking_alg="iou"
    counting_mode="v2"
    mode="video"
    src="${HERE}/data/DTC_counting_test_movie_1_8/"
    src="${HERE}/sample_movie.mp4"
    weights="${HERE}/weights/best.pt"
    project="${HERE}/results"
    save_image="--save_image"
    save_movie="--save_movie"
    device="0"

    python ./src/main.py \
        --tracking_alg "${tracking_alg}" \
        --counting_mode "${counting_mode}" \
        --mode "${mode}" \
        --device ${device} \
        --weights "${weights}" \
        --project "${project}" \
        "${save_image}" \
        "${save_movie}" \
        --source "${src}" 2>&1 |  tee "$LOG_DIR/$which.log.$DATE"
fi


if [ "${which}" == git ]; then
    comment=""
    git add . 
    git commit -m "$(date +%Y.%m.%d) $comment"
    git push
fi


THIS_END=$(date +%Y.%m.%d_%H.%M.%S)
echo "[END  : $which] [$THIS_END] ######################################################################"
