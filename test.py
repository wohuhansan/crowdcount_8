import os
import numpy as np
import cv2
import torch
from src.utils import log, is_only_one_bool_is_true, gray_to_bgr, make_path
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
if not is_only_one_bool_is_true(is_shtech_A, is_shtech_B, is_ucf_cc_50_1, is_ucf_cc_50_2, is_ucf_cc_50_3, is_ucf_cc_50_4, is_ucf_cc_50_5, is_worldexpo_1, is_worldexpo_2, is_worldexpo_3, is_worldexpo_4, is_worldexpo_5, is_airport, is_trancos):
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
    # test_model_path = './saved_models_shtA/shtechA_13.h5'
    test_model_path = r'E:\PycharmProjects\crowdcount_4_density\best_models\model_180629\shanghaitechA\shanghaitech_part_A_patches_9_random_overturn_rgb\mse_loss\test_mae\shtechA_49.h5'
    original_dataset_name = 'shtechA'

    test_data_config = dict()
    test_data_config['shtA1Resize1_test'] = test_flag.copy()
elif is_shtech_B:
    test_model_path = './best_models/model_180629/shanghaitechB/formatted_trainval_15_15/shanghaitech_part_B_patches_9_random_overturn_rgb_more_than_one_pedestrain/block_loss_4/shtechB_7.h5'

    dataset_name = 'shtechB'

    data_path = {}

    data_path['train'] = None

    data_path['validation'] = None

    test_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_B_patches_1_rgb/test'
    blob['gt'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_B_patches_1_rgb/test_den'
    blob['roi'] = None
    test_path.append(blob)
    data_path['test'] = test_path
elif is_ucf_cc_50:
    if is_ucf_cc_50_1:
        i = 1
        test_model_path = './best_models/model_180629/ucf/ucf_cc_50_patches_9_random_rgb_overturn/1/ucf_cc_50_1_1886.h5'
    elif is_ucf_cc_50_2:
        i = 2
        test_model_path = './best_models/model_180629/ucf/ucf_cc_50_patches_9_random_rgb_overturn/2/ucf_cc_50_2_23.h5'
    elif is_ucf_cc_50_3:
        i = 3
        test_model_path = './best_models/model_180629/ucf/ucf_cc_50_patches_9_random_rgb_overturn/3/ucf_cc_50_3_134.h5'
    elif is_ucf_cc_50_4:
        i = 4
        test_model_path = './best_models/model_180629/ucf/ucf_cc_50_patches_9_random_rgb_overturn/4/ucf_cc_50_4_11_1.h5'
    elif is_ucf_cc_50_5:
        i = 5
        test_model_path = './best_models/model_180629/ucf/ucf_cc_50_patches_9_random_rgb_overturn/5/ucf_cc_50_5_1539.h5'

    dataset_name = 'ucf_cc_50_{}'.format(i)

    data_path = {}

    data_path['train'] = None

    data_path['validation'] = None

    test_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/{}/val'.format(i)
    blob['gt'] = '/media/dell/OS/data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/{}/val_den'.format(i)
    blob['roi'] = None
    test_path.append(blob)
    data_path['test'] = test_path
elif is_worldexpo:
    if is_worldexpo_1:
        i = 1
        test_model_path = './best_models/model_180629/worldexpo10/worldexpo_patches_1_rgb_overturn/block_loss_4/worldexpo_1_18.h5'
    elif is_worldexpo_2:
        i = 2
        test_model_path = './best_models/model_180629/worldexpo10/worldexpo_patches_1_rgb_overturn/block_loss_4/worldexpo_2_18.h5'
    elif is_worldexpo_3:
        i = 3
        test_model_path = './best_models/model_180629/worldexpo10/worldexpo_patches_1_rgb_overturn/block_loss_4/worldexpo_3_18.h5'
    elif is_worldexpo_4:
        i = 4
        test_model_path = './best_models/model_180629/worldexpo10/worldexpo_patches_1_rgb_overturn/block_loss_4/worldexpo_4_6.h5'
    elif is_worldexpo_5:
        i = 5
        test_model_path = './best_models/model_180629/worldexpo10/worldexpo_patches_1_rgb_overturn/block_loss_4/worldexpo_5_18.h5'

    dataset_name = 'worldexpo_{}'.format(i)

    data_path = {}

    data_path['train'] = None

    data_path['validation'] = None

    test_path = []

    blob = {}
    blob['image'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/{}'.format(i)
    blob['gt'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/{}'.format(i)
    blob['roi'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/{}'.format(i)
    test_path.append(blob)

    data_path['test'] = test_path
elif is_airport:
    dataset_name = 'airport'

    train_image_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train'
    train_gt_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_den'
    train_roi_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_roi'

    validation_image_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test'
    validation_gt_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_den'
    validation_roi_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_roi'

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
# make_path(os.path.join(save_path, 'estimate_prior'))
# make_path(os.path.join(save_path, 'estimate_prior_merged'))

# calculate error on the test dataset
for data_name in test_data_config:
    data = all_data[data_name]['data']

    MAE = 0.0
    index = 0
    for blob in data:
        image_data = blob['image']
        ground_truth_data = blob['density']
        roi = blob['roi']
        image_name = blob['image_name']

        estimate_map = net(image_data)

        ground_truth_map = ground_truth_data.data.cpu().numpy()
        estimate_map = estimate_map.data.cpu().numpy()

        ground_truth_count = np.sum(ground_truth_map)
        estimate_count = np.sum(estimate_map)
        MAE += np.abs(ground_truth_count - estimate_count)
        index += 1

        # save all kinds of map
        cv2.imwrite(os.path.join(save_path, 'ground_truth_map', '%s_ground_truth_map.jpg' % image_name), gray_to_bgr(ground_truth_map[0][0]))
        cv2.imwrite(os.path.join(save_path, 'estimate_map', '%s_estimate_map.jpg' % image_name), gray_to_bgr(estimate_map[0][0]))

        # alpha = 0.5
        # image = image_data[0]
        # image = np.moveaxis(image, 0, 2).astype(np.uint8)  # reshape (3, h, w) to (h, w, 3), type float32 to uint8
        # height = image.shape[0]
        # width = image.shape[1]
        #
        # estimate_prior_normalized = estimate_prior[0] / np.max(estimate_prior)
        # i = 0
        # for prior in estimate_prior_normalized:
        #     prior_bgr = gray_to_bgr(prior)
        #
        #     image_estimate_prior = cv2.addWeighted(image, alpha, cv2.resize(prior_bgr, (width, height)), 1 - alpha, 0)
        #     cv2.imwrite(os.path.join(save_path, 'estimate_prior', '%s_image_estimate_prior_%d.jpg' % (image_name, i)), image_estimate_prior)
        #
        #     i += 1
        #
        # merged_estimate_prior = np.argmax(estimate_prior, axis=1)[0]
        # merged_estimate_prior = merged_estimate_prior / np.max(merged_estimate_prior)
        #
        # merged_estimate_prior_bgr = gray_to_bgr(merged_estimate_prior)
        #
        # image_estimate_prior = cv2.addWeighted(image, alpha, cv2.resize(merged_estimate_prior_bgr, (width, height)), 1 - alpha, 0)
        # cv2.imwrite(os.path.join(save_path, 'estimate_prior_merged', '%s_image_estimate_prior_merged.jpg' % image_name), image_estimate_prior)

    MAE = MAE / index
    print(MAE)

# log(log_path, log_info)

