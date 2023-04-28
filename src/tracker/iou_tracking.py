# -*- coding: utf-8 -*-
import numpy as np
import cv2
from .trash import Trash
from .fpsrate import FpsWithTick
import os
import sys
if True:
    sys.path.append('../')
    from utils.count_utils import convert_to_latlng


class Iou_Tracker(object):
    def __init__(self, max_age=2, line_down=None, save_image_path=None):
        self.line_down = line_down
        self.cnt_down = 0
        self.frame_count = 0
        self.trashs = []
        self.max_age = int(max_age)
        self.t_id = 0
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.frame_count = 0
        self.fps_count = 0
        self.fpsWithTick = FpsWithTick()
        self.save_image_path = save_image_path

    def intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = (xB - xA + 1) * (yB - yA + 1)
        xinter = (xB - xA + 1)
        yinter = (yB - yA + 1)
        if xinter <= 0 or yinter <= 0:
            iou = 0
            return iou

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou < 0 or iou > 1:
            iou = 0
        return iou

    def match_trash(self, cords):
        if len(cords) == 0:
            return None
        if len(self.trashs) == 0:
            return np.array([(i, int(y)) for i, y in enumerate(np.zeros(len(cords)))])
        trash_ids = []
        ious_all = []
        for i in self.trashs:
            trash_ids.append(i.id)

        for cord in cords:
            ious = []
            for i in self.trashs:
                iou = self.intersection_over_union(i.cords, cord)
                ious.append(iou)

            ious_all.append(ious)
        ious_all = np.array(ious_all)
        trash_ids = np.array(trash_ids)
        sets = []
        true_array = ious_all > 0
        if True in true_array:
            pass
        else:
            return np.array([(i, int(y)) for i, y in enumerate(np.zeros(len(cords)))])

        while (True):
            true_array = ious_all > 0
            if True in true_array:
                pass
            else:
                break
            temp = list(np.unravel_index(ious_all.argmax(), ious_all.shape))
            ious_all[temp[0]] = -1
            ious_all[:, temp[1]] = -1
            sets.append([temp[0], trash_ids[temp[1]]])

        sets = np.array(sets)
        for i in range(len(cords)):
            if len(sets) >= len(cords):
                break
            if i not in sets[:, 0]:
                sets = np.append(sets, [i, 0]).reshape(-1, 2)

        return sets

    def update(self, cords2=np.empty((0, 5)), frame=None):
        sets = self.match_trash(cords2)

        if sets is not None:
            for cord_id, trash_id in sets:
                cord = cords2[cord_id]
                cord = cord.astype("int64")
                x, y, w, h = cord[0], cord[1], cord[2] - \
                    cord[0], cord[3]-cord[1]
                _center = np.array([int(x + (w/2)), int(y + (h*7/8))])

                if trash_id != 0:
                    for i in self.trashs:
                        if i.id == trash_id:
                            i.updateCoords(cord, _center)
                            if i.going_DOWN(self.line_down):
                                if i.state:
                                    self.cnt_down += 1
                                    i.state = False
                                    i.done = True
                                    if self.save_image_path:
                                        self.make_images(frame, i)

                                    # print("ID:", i.id, 'crossed', self.cnt_down)

                elif trash_id == 0 and _center[1] < self.line_down:
                    t = Trash(self.t_id, cord, _center, self.max_age)
                    self.trashs.append(t)
                    self.t_id += 1

        for i in self.trashs:
            i.age += 1
            if i.age > self.max_age:
                i.done = True

            if i.center[1] > self.line_down:
                i.done = True
            if i.done:
                index = self.trashs.index(i)
                self.trashs.pop(index)

        return self.cnt_down

    def make_images(self, frame, i):
        i.state = False
        i.done = True
        cv2.circle(
            frame, (i.center[0], i.center[1]), 3, (0, 0, 126), -1)
        cv2.rectangle(
            frame, (i.cords[0], i.cords[1]), (i.cords[2], i.cords[3]), (0, 252, 124), 2)
        cv2.rectangle(frame, (i.cords[0], i.cords[1] - 20),
                      (i.cords[0] + 170, i.cords[1]), (0, 252, 124), thickness=2)
        cv2.rectangle(frame, (i.cords[0], i.cords[1] - 20),
                      (i.cords[0] + 170, i.cords[1]), (0, 252, 124), -1)
        str_down = 'COUNT:' + str(self.cnt_down)
        cv2.putText(frame, str(
            i.id) + " " + str(i.age), (i.cords[0], i.cords[1] - 5), self.font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (0, self.line_down), (int(

            frame.shape[1]), self.line_down), (255, 0, 0), 2)
        cv2.putText(
            frame, str_down, (10, 70), self.font, 2.5, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(
            frame, str_down, (10, 70), self.font, 2.5, (255, 255, 255), 8, cv2.LINE_AA)
        print("ID:", i.id, 'crossed', self.cnt_down)
        path = self.save_image_path + "_{0:04d}.jpg".format(self.cnt_down)
        cv2.imwrite(path, frame)
