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

if [ $# -gt 1 ]; then
    script=$0
    for which in "$@"; do
        "${script} ${which}"
    done
    exit
fi

if [ "${which}" == get_files ]; then
    START=297
    END=333
    for ((i=START;i<=END;i++)); do
        pb="abc${i}"
        atcoder-tools gen ${pb}
    done
    
fi


if [ "${which}" == change_name ]; then
    src="${HERE}"
    python utils/xutils.py --which "${which}" \
        --src "${src}" 2>&1 |  tee "$LOG_DIR/$which.log.$DATE"
fi


if [ "${which}" == git ]; then
    comment=""
    git add . 
    git commit -m "$(date +%Y.%m.%d) $comment"
    git push
fi


THIS_END=$(date +%Y.%m.%d_%H.%M.%S)
echo "[END  : $which] [$THIS_END] ######################################################################"
