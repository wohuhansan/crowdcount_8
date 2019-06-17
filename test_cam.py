import os
import torch
import numpy as np
import cv2
from src.utils import gray_to_bgr
from src.crowd_count import CrowdCount
from src import network
from src.utils import ndarray_to_tensor

alpha = 0.5

test_model_path = test_model_path = r'./saved_models_shtA/shtechA_31_5217.h5'
net = CrowdCount()
network.load_net(test_model_path, net)

net.cuda()
net.eval()

# cap = cv2.VideoCapture(r'E:\PycharmProjects\data\video\MVI_1582.MOV')
cap = cv2.VideoCapture(r'E:\PycharmProjects\data\video\DJI_0001.MOV')
# cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception('can not connect to camera')

out = cv2.VideoWriter('test_cam_output.mp4', cv2.VideoWriter_fourcc(*'H264'), 30.0, (1920, 1080))

index = 0

# calculate error on the test dataset
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        index += 1
        if index % 100 == 0:
            print(index)
        # if index < 7.5 * 60 * 30:
        #     continue

        original_height = frame.shape[0]
        original_width = frame.shape[1]

        process_height = 1080
        process_width = int(original_width / original_height * process_height)

        TIMES = 16
        process_height = int(process_height / TIMES) * TIMES
        process_width = int(process_width / TIMES) * TIMES

        frame = cv2.resize(frame, (process_width, process_height))

        # get original size
        height = frame.shape[0]
        width = frame.shape[1]

        reshaped_frame = np.moveaxis(frame, 2, 0).astype(np.float32)  # reshape (h, w, 3) to (3, h, w)
        reshaped_frame = reshaped_frame.reshape((1, 3, height, width))

        image_data = ndarray_to_tensor(reshaped_frame, is_cuda=True)
        estimate_map, _ = net(image_data)
        estimate_map = estimate_map.data.cpu().numpy()

        estimate_count = np.sum(estimate_map)

        max_value = np.max(estimate_map)
        if max_value > 0:
            estimate_prior_normalized = estimate_map[0][0] / np.max(estimate_map)
        else:
            estimate_prior_normalized = estimate_map[0][0]

        estimate_prior_normalized_bgr = gray_to_bgr(estimate_prior_normalized)

        image_estimate_map = cv2.addWeighted(frame, alpha, cv2.resize(estimate_prior_normalized_bgr, (width, height)), 1 - alpha, 0)

        estimate_count_text = '%.f' % estimate_count
        t_size = cv2.getTextSize(estimate_count_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.putText(image_estimate_map, estimate_count_text, (0, t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 1)

        image_estimate_map = cv2.resize(image_estimate_map, (1920, 1080))
        out.write(image_estimate_map)

        cv2.imshow("cam", image_estimate_map)
        cv2.waitKey(1)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
