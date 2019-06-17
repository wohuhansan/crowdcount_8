import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
import openpyxl as excel

from src.utils import log, is_only_one_bool_is_true
from src.data import Data

# training configuration
max_epoch = 100 # maximum training times
lr_adjust_epoch = None # lr * 0.1 every lr_adjust_epoch steps
lr = 0.00001 # default 0.00001
# random_seed = 64678
random_seed = None

is_use_tensorboard = False # use tensorflow tensorboard

is_load_pretrained_model = True # load parameters of fine-tuned model

# choose validation method
is_use_validation = False # use validation set to choose best model
is_use_train_mae = False # ignore validation set and use train error to choose best model
is_use_test = True # use test set to choose best model
if not is_only_one_bool_is_true(is_use_validation, is_use_train_mae, is_use_test):
    raise Exception('only one validation set should be chosen')

is_use_mae = True # use mean absolute error to choose best model
is_use_game = False # use grid average mean absolute error to choose best model
if not is_only_one_bool_is_true(is_use_mae, is_use_game):
    raise Exception('only one validation method should be chosen')

is_random_flip = False # randomly filp train image
is_random_noise = False # randomly add noise to train image

is_pre_load_data = False

# choose one dataset
is_shtech_A = True
is_shtech_B = False
is_ucf_cc_50_1 = False
is_ucf_cc_50_2 = False
is_ucf_cc_50_3 = False
is_ucf_cc_50_4 = False
is_ucf_cc_50_5 = False
is_worldexpo = False
is_airport = False
is_ucsd = False
is_trancos = False
is_mall = False
if not is_only_one_bool_is_true(is_shtech_A, is_shtech_B, is_ucf_cc_50_1, is_ucf_cc_50_2, is_ucf_cc_50_3, is_ucf_cc_50_4, is_ucf_cc_50_5, is_worldexpo, is_airport, is_ucsd, is_trancos, is_mall):
    raise Exception('only one dataset should be chosen')

if is_use_tensorboard:
    summary_writer = SummaryWriter()
if is_load_pretrained_model:
    finetune_model_path = []
    finetune_model_path.append('./pretrained_vgg16.h5')
if is_shtech_A or is_shtech_B:
    is_shtech = True
else:
    is_shtech = False
if is_ucf_cc_50_1 or is_ucf_cc_50_2 or is_ucf_cc_50_3 or is_ucf_cc_50_4 or is_ucf_cc_50_5:
    is_ucf_cc_50 = True
else:
    is_ucf_cc_50 = False

log_path = "log.txt"

if is_shtech_A:
    dataset_name = 'shtechA'
    output_dir = './saved_models_shtA/'

    data_path = {}

    blob = {}
    # blob['image'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train'
    # blob['gt'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train_den'
    blob['image'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/test'
    blob['gt'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/test_den'
    blob['roi'] = None
    data_path['train'] = blob

    # validation_path = []
    # blob = {}
    # blob['image'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/train'
    # blob['gt'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/train_den'
    # blob['roi'] = None
    # validation_path.append(blob)
    # data_path['validation'] = validation_path
    data_path['validation'] = None

    # test_path = []
    # blob = {}
    # blob['image'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/test'
    # blob['gt'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/test_den'
    # blob['roi'] = None
    # test_path.append(blob)
    # data_path['test'] = test_path
    data_path['test'] = None
elif is_shtech_B:
    dataset_name = 'shtechB'
    output_dir = './saved_models_shtB/'

    data_path = {}

    blob = {}
    blob['image'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_9_random_overturn_rgb_more_than_one_pedestrain/train'
    blob['gt'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_9_random_overturn_rgb_more_than_one_pedestrain/train_den'
    blob['roi'] = None
    data_path['train'] = blob

    validation_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_B_patches_1_rgb/train'
    blob['gt'] = '/media/dell/OS/data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_B_patches_1_rgb/train_den'
    blob['roi'] = None
    validation_path.append(blob)
    data_path['validation'] = validation_path

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
    elif is_ucf_cc_50_2:
        i = 2
    elif is_ucf_cc_50_3:
        i = 3
    elif is_ucf_cc_50_4:
        i = 4
    elif is_ucf_cc_50_5:
        i = 5

    dataset_name = 'ucf_cc_50_{}'.format(i)
    output_dir = './saved_models_ucf_{}/'.format(i)

    data_path = {}

    blob = {}
    blob['image'] = '/media/dell/OS/data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/{}/train'.format(i)
    blob['gt'] = '/media/dell/OS/data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/{}/train_den'.format(i)
    blob['roi'] = None
    data_path['train'] = blob

    validation_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/{}/train'.format(i)
    blob['gt'] = '/media/dell/OS/data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/{}/train_den'.format(i)
    blob['roi'] = None
    validation_path.append(blob)
    data_path['validation'] = validation_path

    test_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/{}/val'.format(i)
    blob['gt'] = '/media/dell/OS/data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/{}/val_den'.format(i)
    blob['roi'] = None
    test_path.append(blob)
    data_path['test'] = test_path
elif is_worldexpo:
    dataset_name = 'worldexpo_all'
    output_dir = './saved_models_worldexpo_all/'

    data_path = {}

    blob = {}
    blob['image'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train'
    blob['gt'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train_den'
    blob['roi'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train_roi'
    data_path['train'] = blob

    # validation_path = []
    # blob = {}
    # blob['image'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train'
    # blob['gt'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train_den/'
    # blob['roi'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train_roi/'
    # validation_path.append(blob)
    # data_path['validation'] = validation_path
    data_path['validation'] = None

    test_path = []
    for i in range(1, 6):
        blob = {}
        blob['image'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/' + str(i)
        blob['gt'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/' + str(i)
        blob['roi'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/' + str(i)
        test_path.append(blob)
    blob = {}
    blob['image'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/all'
    blob['gt'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/all'
    blob['roi'] = '/media/dell/OS/data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all'
    test_path.append(blob)
    data_path['test'] = test_path
elif is_airport:
    dataset_name = 'airport'
    output_dir = './saved_models_airport/'

    train_image_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train'
    train_gt_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_den'
    train_roi_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_roi'

    validation_image_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test'
    validation_gt_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_den'
    validation_roi_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_roi'
elif is_ucsd:
    dataset_name = 'ucsd'
    output_dir = './saved_models_ucsd/'

    data_path = {}

    blob = {}
    blob['image'] = '/media/dell/OS/data/ucsd/formatted_trainval_15_4/ucsd_patches_1_overturn_resize_4_rgb/train'
    blob['gt'] = '/media/dell/OS/data/ucsd/formatted_trainval_15_4/ucsd_patches_1_overturn_resize_4_rgb/train_den'
    blob['roi'] = '/media/dell/OS/data/ucsd/formatted_trainval_15_4/ucsd_patches_1_overturn_resize_4_rgb/train_roi'
    data_path['train'] = blob

    # validation_path = []
    # blob = {}
    # blob['image'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train'
    # blob['gt'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train_den/'
    # blob['roi'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train_roi/'
    # validation_path.append(blob)
    # data_path['validation'] = validation_path
    data_path['validation'] = None

    test_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/ucsd/formatted_trainval_15_4/ucsd_patches_1_resize_4_rgb/test'
    blob['gt'] = '/media/dell/OS/data/ucsd/formatted_trainval_15_4/ucsd_patches_1_resize_4_rgb/test_den'
    blob['roi'] = '/media/dell/OS/data/ucsd/formatted_trainval_15_4/ucsd_patches_1_resize_4_rgb/test_roi'
    test_path.append(blob)
    data_path['test'] = test_path
elif is_trancos:
    dataset_name = 'trancos'
    output_dir = './saved_models_trancos/'

    data_path = {}

    blob = {}
    blob['image'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn'
    blob['gt'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn_den'
    blob['roi'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn_roi'
    data_path['train'] = blob

    validation_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val'
    blob['gt'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val_den/'
    blob['roi'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val_roi/'
    validation_path.append(blob)
    data_path['validation'] = validation_path

    test_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test'
    blob['gt'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test_den'
    blob['roi'] = '/media/dell/OS/data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test_roi'
    test_path.append(blob)
    data_path['test'] = test_path
elif is_mall:
    dataset_name = 'mall'
    output_dir = './saved_models_mall/'

    data_path = {}

    blob = {}
    blob['image'] = '/media/dell/OS/data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train'
    blob['gt'] = '/media/dell/OS/data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train_den'
    blob['roi'] = '/media/dell/OS/data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train_roi'
    data_path['train'] = blob

    # validation_path = []
    # blob = {}
    # blob['image'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train'
    # blob['gt'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train_den/'
    # blob['roi'] = '../data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/train_roi/'
    # validation_path.append(blob)
    # data_path['validation'] = validation_path
    data_path['validation'] = None

    test_path = []
    blob = {}
    blob['image'] = '/media/dell/OS/data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val'
    blob['gt'] = '/media/dell/OS/data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val_den'
    blob['roi'] = '/media/dell/OS/data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val_roi'
    test_path.append(blob)
    data_path['test'] = test_path

if random_seed is not None:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

# load data
data = Data(data_path, shuffle=True, random_seed=random_seed, pre_load=is_pre_load_data, is_label=False)
data = data.get()

save_path = './info'

if not os.path.exists(save_path):
    os.mkdir(save_path)

excel_book = excel.Workbook()
excel_sheet = excel_book.active
excel_sheet.title = 'Image'
excel_sheet['A1'] = 'filename'
excel_sheet['B1'] = 'gt max density'
excel_sheet['C1'] = 'gt mean density'
row_count = 1

train_data = data['train']

for blob in train_data:
    image_data = blob['image']
    gt_data = blob['density_map']
    roi = blob['roi']
    filename = blob['filename']

    row_count += 1
    excel_sheet['A' + str(row_count)] = filename
    excel_sheet['B' + str(row_count)] = np.max(gt_data)
    excel_sheet['C' + str(row_count)] = np.mean(gt_data)

    if row_count % 100 == 0:
        print('row %d: done' % row_count)

excel_book.save(os.path.join(save_path, 'dataset_info.xlsx'))