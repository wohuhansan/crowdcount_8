import os
import torch
import cv2
import numpy as np
import time

from src.utils import log, is_only_one_bool_is_true, gray_to_bgr, make_path, calculate_game
from src.crowd_count import CrowdCount
from src.data import Data
from src import network

# choose one dataset
is_shtech_A = True
is_shtech_B = False
is_ucf_cc_50_1 = False
is_ucf_cc_50_2 = False
is_ucf_cc_50_3 = False
is_ucf_cc_50_4 = False
is_ucf_cc_50_5 = False
is_worldexpo_1 = False
is_worldexpo_2 = False
is_worldexpo_3 = False
is_worldexpo_4 = False
is_worldexpo_5 = False
is_airport = False
is_trancos = False
is_ucf_qnrf = False
if not is_only_one_bool_is_true(is_shtech_A, is_shtech_B, is_ucf_cc_50_1, is_ucf_cc_50_2, is_ucf_cc_50_3, is_ucf_cc_50_4, is_ucf_cc_50_5, is_worldexpo_1, is_worldexpo_2, is_worldexpo_3, is_worldexpo_4, is_worldexpo_5, is_airport, is_trancos, is_ucf_qnrf):
    raise Exception('only one dataset should be chosen')

if is_shtech_A or is_shtech_B:
    is_shtech = True
else:
    is_shtech = False
if is_ucf_cc_50_1 or is_ucf_cc_50_2 or is_ucf_cc_50_3 or is_ucf_cc_50_4 or is_ucf_cc_50_5:
    is_ucf_cc_50 = True
else:
    is_ucf_cc_50 = False
if is_worldexpo_1 or is_worldexpo_2 or is_worldexpo_3 or is_worldexpo_4 or is_worldexpo_5:
    is_worldexpo = True
else:
    is_worldexpo = False

log_path = "log_test.txt"

test_flag = dict()
test_flag['preload'] = False
test_flag['label'] = False
test_flag['mask'] = False

if is_shtech_A:
    # test_model_path = r'./saved_models_shtA/shtechA_41_5399.h5'
    test_model_path = r'E:\PycharmProjects\crowdcount_8\best_models\model_20190520_mapMaskScaleMutiple_3maskClassifier64\0_40_9patch_62.04_99.57/saved_models_shtA/shtechA_19_5399.h5'
    original_dataset_name = 'shtechA'

    test_data_config = dict()
    test_data_config['shtA1_test'] = test_flag.copy()
elif is_shtech_B:
    test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\baseline\mcnn_shtechB_15.h5'
    # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\shanghaiB_9.65\mcnn_shtechB_26.h5'
    original_dataset_name = 'shtechB'

    test_data_config = dict()
    test_data_config['shtB1_test'] = test_flag.copy()
elif is_ucf_cc_50:
    original_dataset_name = 'ucf'
    test_data_config = dict()
    if is_ucf_cc_50_1:
        test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\baseline\mcnn_ucf_cc_50_1_13.h5'
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\ucf_244\1_155_0.0005_4\mcnn_ucf_cc_50_1_1.h5'
        test_data_config['ucf1_test1'] = test_flag.copy()
    elif is_ucf_cc_50_2:
        test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\baseline\mcnn_ucf_cc_50_2_32.h5'
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\ucf_244\2_208_0.0001_4\mcnn_ucf_cc_50_2_9.h5'
        test_data_config['ucf1_test2'] = test_flag.copy()
    elif is_ucf_cc_50_3:
        test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\baseline\mcnn_ucf_cc_50_3_13.h5'
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\ucf_244\3_281_0.0003_4\mcnn_ucf_cc_50_3_45.h5'
        test_data_config['ucf1_test3'] = test_flag.copy()
    elif is_ucf_cc_50_4:
        test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\baseline\mcnn_ucf_cc_50_4_7.h5'
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\ucf_244\4_224_0.0005_4\mcnn_ucf_cc_50_4_46.h5'
        test_data_config['ucf1_test4'] = test_flag.copy()
    elif is_ucf_cc_50_5:
        test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\baseline\mcnn_ucf_cc_50_5_38.h5'
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\ucf_244\5_355_0.0001_4\mcnn_ucf_cc_50_5_0.h5'
        test_data_config['ucf1_test5'] = test_flag.copy()
elif is_worldexpo:
    original_dataset_name = 'worldexpo'
    test_data_config = dict()

    test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\baseline\mcnn_worldexpo_all_19.h5'
    if is_worldexpo_1:
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\worldExpo10_6.5\mcnn_worldexpo_all_5.h5'
        test_data_config['we1_test1'] = test_flag.copy()
    elif is_worldexpo_2:
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\worldExpo10_6.5\mcnn_worldexpo_all_9.h5'
        test_data_config['we1_test2'] = test_flag.copy()
    elif is_worldexpo_3:
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\worldExpo10_6.5\mcnn_worldexpo_all_12.h5'
        test_data_config['we1_test3'] = test_flag.copy()
    elif is_worldexpo_4:
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\worldExpo10_6.5\mcnn_worldexpo_all_7.h5'
        test_data_config['we1_test4'] = test_flag.copy()
    elif is_worldexpo_5:
        # test_model_path = r'E:\PycharmProjects\final_models\DensityCNN\worldExpo10_6.5\mcnn_worldexpo_all_12.h5'
        test_data_config['we1_test5'] = test_flag.copy()
elif is_airport:
    dataset_name = 'airport'

    train_image_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train'
    train_gt_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_den'
    train_roi_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_roi'

    validation_image_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test'
    validation_gt_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_den'
    validation_roi_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_roi'
elif is_trancos:
    test_model_path = r'E:\PycharmProjects\crowdcount_8\best_models\model_20190318_DPLNet\trancos\pool_16_3.83\saved_models_trancos\trancos_18.h5'
    # test_model_path = './saved_models_trancos/trancos_9.h5'
    original_dataset_name = 'trancos'

    test_data_config = dict()
    test_data_config['tran1Resize1_test'] = test_flag.copy()
elif is_ucf_qnrf:
    test_model_path = r'E:\PycharmProjects\crowdcount_8\best_models\model_20190318_DPLNet\ucf_qnrf\pool_4_112.33_205.77\saved_models_ucf_qnrf\ucf_qnrf_26_21617.h5'
    original_dataset_name = 'ucf_qnrf'

    test_data_config = dict()
    test_data_config['ucfQnrf1Resize1024_test'] = test_flag.copy()

# load data
all_data = Data(test_data_config)
all_data = all_data.get()

net = CrowdCount()

network.load_net(test_model_path, net)

net.cuda()
net.eval()

# log_info = []

save_path = './test_output'
make_path(save_path)
make_path(os.path.join(save_path, 'ground_truth_map'))
make_path(os.path.join(save_path, 'estimate_map'))
make_path(os.path.join(save_path, 'estimate_raw_map'))
make_path(os.path.join(save_path, 'estimate_scaled_map'))
make_path(os.path.join(save_path, 'estimate_prior_merged'))

total_forward_time = 0.0

# calculate error on the test dataset
for data_name in test_data_config:
    data = all_data[data_name]['data']

    mae = 0.0
    mse = 0.0
    game_0 = 0.0
    game_1 = 0.0
    game_2 = 0.0
    game_3 = 0.0
    index = 0
    for blob in data:
        image_data = blob['image']
        ground_truth_data = blob['density']
        roi = blob['roi']
        image_name = blob['image_name']

        start_time = time.clock()
        estimate_map, visual_dict = net(image_data, roi=roi)
        total_forward_time += time.clock() - start_time

        ground_truth_map = ground_truth_data.data.cpu().numpy()
        estimate_map = estimate_map.data.cpu().numpy()

        ground_truth_count = np.sum(ground_truth_map)
        estimate_count = np.sum(estimate_map)

        mae += np.abs(ground_truth_count - estimate_count)
        mse += (ground_truth_count - estimate_count) ** 2
        game_0 += calculate_game(ground_truth_map, estimate_map, 0)
        game_1 += calculate_game(ground_truth_map, estimate_map, 1)
        game_2 += calculate_game(ground_truth_map, estimate_map, 2)
        game_3 += calculate_game(ground_truth_map, estimate_map, 3)
        index += 1

        # estimate_score_map = visual_dict['score'].data.cpu().numpy()
        # estimate_class_map = visual_dict['class'].data.cpu().numpy()

        # save ground truth and estimate map
        max_value = max(np.max(ground_truth_map), np.max(estimate_map))
        ground_truth_map_normalized = ground_truth_map[0][0] / max_value
        estimate_map_normalized = estimate_map[0][0] / max_value
        cv2.imwrite(os.path.join(save_path, 'ground_truth_map', '%s_ground_truth_map_%.2f.jpg' % (image_name, ground_truth_count)), gray_to_bgr(ground_truth_map_normalized))
        cv2.imwrite(os.path.join(save_path, 'estimate_map', '%s_estimate_map_%.2f.jpg' % (image_name, estimate_count)), gray_to_bgr(estimate_map_normalized))

        # save raw maps and scaled map
        raw_estimate_maps = visual_dict['raw_maps'].data.cpu().numpy()
        scaled_estimate_maps = visual_dict['scaled_maps'].data.cpu().numpy()
        max_value = max(np.max(raw_estimate_maps), np.max(scaled_estimate_maps))
        raw_estimate_maps_normalized = raw_estimate_maps[0] / max_value
        scaled_estimate_maps_normalized = scaled_estimate_maps[0] / max_value
        for i in range(raw_estimate_maps.shape[1]):
            cv2.imwrite(os.path.join(save_path, 'estimate_raw_map', '%s_estimate_raw_map_%d.jpg' % (image_name, i)), gray_to_bgr(raw_estimate_maps_normalized[i]))
        for i in range(scaled_estimate_maps.shape[1]):
            cv2.imwrite(os.path.join(save_path, 'estimate_scaled_map', '%s_estimate_scaled_map_%d.jpg' % (image_name, i)), gray_to_bgr(scaled_estimate_maps_normalized[i]))

        # group = 5
        # if estimate_score_map.shape[1] % group != 0:
        #     raise Exception('invalid group')
        # max_score_array = np.array([np.max(map) for map in estimate_score_map[0]])
        # max_score_array = np.max(max_score_array.reshape(group, -1), axis=1)
        # number = estimate_score_map.shape[1] / group  # number of maps in each group
        # i = 0
        # for map in estimate_score_map[0]:
        #     map_bgr = gray_to_bgr(map / max_score_array[int(i / number)])
        #     cv2.imwrite(os.path.join(save_path, 'estimate_score_map', '%s_image_estimate_score_map_%d.jpg' % (image_name, i)), map_bgr)
        #     i += 1
        #
        # estimate_class_map_normalized = estimate_class_map[0] / np.max(estimate_class_map)
        # i = 0
        # for map in estimate_class_map_normalized:
        #     map_bgr = gray_to_bgr(map)
        #     cv2.imwrite(os.path.join(save_path, 'estimate_class_map', '%s_image_estimate_class_map_%d.jpg' % (image_name, i)), map_bgr)
        #     i += 1
        #
        # alpha = 0.5
        # image = image_data[0].data.cpu().numpy()
        # image = np.moveaxis(image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
        # height = image.shape[0]
        # width = image.shape[1]
        #
        # merged_estimate_class_map = np.argmax(estimate_class_map, axis=1)[0]
        # merged_estimate_class_map = merged_estimate_class_map / (estimate_class_map.shape[1] - 1)
        #
        # merged_estimate_class_map_bgr = gray_to_bgr(merged_estimate_class_map)
        # image_estimate_class_map = cv2.addWeighted(image, alpha, cv2.resize(merged_estimate_class_map_bgr, (width, height)), 1 - alpha, 0)
        # cv2.imwrite(os.path.join(save_path, 'estimate_class_map', '%s_image_estimate_class_map_merged.jpg' % image_name), image_estimate_class_map)

        estimate_prior = visual_dict['masks'].data.cpu().numpy()

        alpha = 0.5
        image = image_data[0].data.cpu().numpy()
        image = np.moveaxis(image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
        height = image.shape[0]
        width = image.shape[1]

        merged_estimate_prior = np.argmax(estimate_prior, axis=1)[0]
        merged_estimate_prior_normalized = merged_estimate_prior / (estimate_prior.shape[1] - 1)

        merged_estimate_prior_bgr = gray_to_bgr(merged_estimate_prior_normalized)
        image_estimate_prior = cv2.addWeighted(image, alpha, cv2.resize(merged_estimate_prior_bgr, (width, height)), 1 - alpha, 0)
        cv2.imwrite(os.path.join(save_path, 'estimate_prior_merged', '%s_image_estimate_prior_merged.jpg' % image_name), image_estimate_prior)

    mae = mae / index
    mse = np.sqrt(mse / index)
    game_0 = game_0 / index
    game_1 = game_1 / index
    game_2 = game_2 / index
    game_3 = game_3 / index
    print('mae: %.2f mse: %.2f game: %.2f %.2f %.2f %.2f' % (mae, mse, game_0, game_1, game_2, game_3))

print('total forward time is %f seconds of %d samples.' % (total_forward_time, index))
# log(log_path, log_info)

