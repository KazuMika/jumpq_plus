# DeepCounter v2

## ディレクトリ構成(主なファイルのみ)

src
├─tracker(tracking と counting を行う:iou と sort のそれぞれ２つの手法)
│ ├iou_tracking.py(iou による tracking)
│ ├trash.py(iou による tracking する時必要になる trash クラス)
│ └─sort.py(sort による tracking)
├utils
│ └─fpsrate.py(FPS を測定する時使用するクラス)
├yolov5(detection を行う)
│ └─models(model が定義されているディレクトリ)
└── counter.py(実際にゴミ袋をカウントしていく Counter クラス)
├── eval.sh(eval.py を実行するシェルスクリプト)
├── eval.py(counter.py を実行するファイル)
└── test.py(yolov5 を 精度(mAP)評価するファイル)
└── train.py(yolov5 を学習するファイル)
