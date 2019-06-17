import threading
import time
from datetime import datetime
import pymysql

import os
import numpy as np
import cv2
import torch
from src.utils import gray_to_bgr
from src.crowd_count import CrowdCount
from src import network
from src.utils import ndarray_to_tensor


class UpdateThred(threading.Thread):
    def __init__(self, num, time_now):
        threading.Thread.__init__(self)
        self.num = num
        self.time_now = time_now

    def run(self):
        db_update(self.num, self.time_now)
        print('OK')


def db_update(num, time_now):
    db = pymysql.connect('nm6012.xyz', 'nm6012_xyz', 'jCmy8ywCWtPWX4LF', 'nm6012_xyz')

    cursor = db.cursor()

    sql = "UPDATE CrowdCount SET num=%i,time='%s' WHERE id=1" % (num, time_now)

    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()

    db.close()


class Predictor:
    def __init__(self):
        test_model_path = r'E:\PycharmProjects\crowdcount_8\best_models\model_20190318_DPLNet\shanghaitechA\pool_4_64.06\saved_models_shtA\shtechA_13.h5'
        # test_model_path = r'.\best_models\model_20190318_DPLNet\trancos\pool_4_3.08_4.39_5.79_7.62\saved_models_trancos\trancos_889.h5'

        self.alpha = 0.5

        self.net = CrowdCount()
        network.load_net(test_model_path, self.net)
        self.net.cuda()
        self.net.eval()

    def predict(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]

        process_height = 400
        process_width = int(width / height * process_height)

        frame = cv2.resize(frame, (process_width, process_height))

        # get original size
        height = frame.shape[0]
        width = frame.shape[1]

        reshaped_frame = np.moveaxis(frame, 2, 0).astype(np.float32)  # reshape (h, w, 3) to (3, h, w)
        reshaped_frame = reshaped_frame.reshape((1, 3, height, width))

        image_data = ndarray_to_tensor(reshaped_frame, is_cuda=True)
        estimate_map, _ = self.net(image_data)
        estimate_map = estimate_map.data.cpu().numpy()

        estimate_count = np.sum(estimate_map)

        max_value = np.max(estimate_map)
        if max_value > 0:
            estimate_prior_normalized = estimate_map[0][0] / np.max(estimate_map)
        else:
            estimate_prior_normalized = estimate_map[0][0]

        estimate_prior_normalized_bgr = gray_to_bgr(estimate_prior_normalized)

        image_estimate_map = cv2.addWeighted(frame, self.alpha, cv2.resize(estimate_prior_normalized_bgr, (width, height)), 1 - self.alpha, 0)

        estimate_count_text = '%.f' % estimate_count
        t_size = cv2.getTextSize(estimate_count_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.putText(image_estimate_map, estimate_count_text, (0, t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 1)

        return image_estimate_map, estimate_count


if __name__ == '__main__':
    last_update_time = time.clock()
    predictor = Predictor()

    cap = cv2.VideoCapture(r'E:\PycharmProjects\data\video\Monkstown.mp4')
    # cap = cv2.VideoCapture(r'E:\PycharmProjects\data\video\traffic\MVI_1582_1585.MOV')
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception('can not connect to camera')

    # calculate error on the test dataset
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            density_map, count = predictor.predict(frame)

            cv2.imshow("cam", density_map)
            cv2.waitKey(1)

            if time.clock() - last_update_time > 2:
                now = datetime.now()
                time_now = '%.4i-%.2i-%.2i %.2i:%.2i:%.2i' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
                thread = UpdateThred(count, time_now)
                thread.start()
                last_update_time = time.clock()
        else:
            break
