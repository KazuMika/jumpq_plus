# -*- coding: utf-8 -*-

def make_movie_and_images(self, p, im0, dets_results, conf_results):
    save_path = str(self.save_movies_dir / p.name)  # img.jpg
    if self.vid_path != save_path and self.save_movie:  # new video
        self.vid_path = save_path
        if isinstance(self.vid_writer, cv2.VideoWriter):
            self.vid_writer.release()  # release previous video writer
        if self.vid_cap:  # video
            fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:  # stream
            fps, w, h = 30, im0.shape[1], im0.shape[0]
        save_vid_path = save_path + '.mp4'
        self.vid_writer = cv2.VideoWriter(save_vid_path,
                                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    str_down = 'COUNT:' + str(self.cnt_down)

    cv2.line(im0, (0, self.line_down),
             (int(im0.shape[1]), self.line_down), (255, 0, 0), 2)
    cv2.putText(im0, str_down, (10, 70), self.font,
                2.0, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(im0, str_down, (10, 70), self.font,
                2.0, (255, 255, 255), 8, cv2.LINE_AA)

    for d, conf in zip(dets_results, conf_results):
        center_x = (d[0] + d[2]) // 2
        center_y = (d[1] + d[3]) // 2
        # if self.line_down >= center_y:
        cv2.circle(im0, (center_x, center_y), 3, (0, 0, 126), -1)
        cv2.rectangle(
            im0, (d[0], d[1]), (d[2], d[3]), (0, 252, 124), 2)

        cv2.rectangle(im0, (d[0], d[1] - 20),
                      (d[0] + 60, d[1]), (0, 252, 124), thickness=2)
        cv2.rectangle(im0, (d[0], d[1] - 20),
                      (d[0] + 60, d[1]), (0, 252, 124), -1)
        cv2.putText(im0, str(int(conf.item() * 100)) + '%',
                    (d[0], d[1] - 5), self.font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    if self.save_movie:
        self.vid_writer.write(im0)
    if self.cnt_down != self.pre_cnt_down:
        try:
            save_img_path = str(self.save_images_dir / p.name)  # img.jpg
            save_image_path = save_img_path + '_' + \
                str(self.number_exp).zfill(4) + '_' + \
                str(self.cnt_down).zfill(4) + '.jpg'
            cv2.imwrite(save_image_path, im0)
            self.image_save_stack.append([save_image_path, im0])
            self.pre_cnt_down = self.cnt_down
        except Exception as e:
            print(e)
