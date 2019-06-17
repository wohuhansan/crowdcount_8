import os
import torch
import numpy as np
import sys
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

from src.utils import log, is_only_one_bool_is_true, ExcelLog, compare_result
from src.crowd_count import CrowdCount
from src import network
from src.data import Data
from src.evaluate_model import evaluate_model
from src.data_multithread_preload import multithread_dataloader

if __name__ == '__main__':
    torch.cuda.set_device(0)

    # training configuration
    max_epoch = 1000  # maximum training times
    lr_adjust_epoch = None  # lr * 0.1 every lr_adjust_epoch steps
    lr = 0.00001  # default 0.00001
    # random_seed = 64678
    random_seed = None
    train_batch_size = 1

    # save models in one epoch
    # 0 means dont save model in epoch, number above 0 means the number of models saved every epoch
    is_save_model_in_epoch = False
    steps_to_save_model = 5405

    is_use_tensorboard = True  # use tensorflow tensorboard

    is_load_pretrained_model = True  # load parameters of fine-tuned model

    key_error = 'mae'
    # key_error = 'game_0'

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
    is_ucf_qnrf = False
    if not is_only_one_bool_is_true(is_shtech_A, is_shtech_B, is_ucf_cc_50_1, is_ucf_cc_50_2, is_ucf_cc_50_3, is_ucf_cc_50_4, is_ucf_cc_50_5, is_worldexpo, is_airport, is_ucsd, is_trancos, is_mall, is_ucf_qnrf):
        raise Exception('only one dataset should be chosen')

    if is_use_tensorboard:
        summary_writer = SummaryWriter()
    if is_load_pretrained_model:
        pretrained_model_path = list()
        pretrained_model_path.append('./pretrained_models/model_201905182237.h5')
        pretrained_model_path.append('../pretrained_vgg16.h5')
        # pretrained_model_path.append('/home/dell/PycharmProjects/zhangli/pretrained_vgg16_12_layers.h5')
    if is_shtech_A or is_shtech_B:
        is_shtech = True
    else:
        is_shtech = False
    if is_ucf_cc_50_1 or is_ucf_cc_50_2 or is_ucf_cc_50_3 or is_ucf_cc_50_4 or is_ucf_cc_50_5:
        is_ucf_cc_50 = True
    else:
        is_ucf_cc_50 = False

    log_path = "log.txt"

    train_flag = dict()
    train_flag['preload'] = False
    train_flag['label'] = False
    train_flag['mask'] = False
    train_flag['shuffle'] = True
    train_flag['seed'] = random_seed
    train_flag['batch_size'] = train_batch_size

    evaluate_flag = dict()
    evaluate_flag['preload'] = False

    # do not preload data when debugging
    if sys.gettrace() is not None:
        train_flag['preload'] = False
        evaluate_flag['preload'] = False

    train_data_config = dict()
    validation_data_config = dict()
    test_data_config = dict()

    if is_shtech_A:
        original_dataset_name = 'shtechA'
        output_dir = './saved_models_shtA/'

        train_data_config['shtA9RandomOverturn_train'] = train_flag.copy()
        # train_data_config['shtA0RandomOverturn_128_256_train'] = train_flag.copy()
        validation_data_config['shtA1_train'] = evaluate_flag.copy()
        test_data_config['shtA1_test'] = evaluate_flag.copy()
    elif is_shtech_B:
        original_dataset_name = 'shtechB'
        output_dir = './saved_models_shtB/'

        train_data_config['shtB9RandFlip_train'] = train_flag.copy()
        validation_data_config['shtB1_train'] = evaluate_flag.copy()
        test_data_config['shtB1_test'] = evaluate_flag.copy()
    elif is_ucf_cc_50:
        if is_ucf_cc_50_1:
            original_dataset_name = 'ucf_cc_50_1'
            output_dir = './saved_models_ucf_1/'

            train_data_config['ucf9RandFlip_train1'] = train_flag.copy()
            test_data_config['ucf1_test1'] = evaluate_flag.copy()
        elif is_ucf_cc_50_2:
            original_dataset_name = 'ucf_cc_50_2'
            output_dir = './saved_models_ucf_2/'

            train_data_config['ucf9RandFlip_train2'] = train_flag.copy()
            test_data_config['ucf1_test2'] = evaluate_flag.copy()
        elif is_ucf_cc_50_3:
            original_dataset_name = 'ucf_cc_50_3'
            output_dir = './saved_models_ucf_3/'

            train_data_config['ucf9RandFlip_train3'] = train_flag.copy()
            test_data_config['ucf1_test3'] = evaluate_flag.copy()
        elif is_ucf_cc_50_4:
            original_dataset_name = 'ucf_cc_50_4'
            output_dir = './saved_models_ucf_4/'

            train_data_config['ucf9RandFlip_train4'] = train_flag.copy()
            test_data_config['ucf1_test4'] = evaluate_flag.copy()
        elif is_ucf_cc_50_5:
            original_dataset_name = 'ucf_cc_50_5'
            output_dir = './saved_models_ucf_5/'

            train_data_config['ucf9RandFlip_train5'] = train_flag.copy()
            test_data_config['ucf1_test5'] = evaluate_flag.copy()
    elif is_worldexpo:
        original_dataset_name = 'worldexpo_all'
        output_dir = './saved_models_worldexpo_all/'

        train_data_config['we1Flip_train'] = train_flag.copy()
        # validation_data_config['shtB1_train'] = evaluate_flag.copy()
        test_data_config['we1_test1'] = evaluate_flag.copy()
        test_data_config['we1_test2'] = evaluate_flag.copy()
        test_data_config['we1_test3'] = evaluate_flag.copy()
        test_data_config['we1_test4'] = evaluate_flag.copy()
        test_data_config['we1_test5'] = evaluate_flag.copy()
    elif is_airport:
        original_dataset_name = 'airport'
        output_dir = './saved_models_airport/'

        train_image_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train'
        train_gt_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_den'
        train_roi_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_roi'

        validation_image_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test'
        validation_gt_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_den'
        validation_roi_path = './data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_roi'
    elif is_ucsd:
        original_dataset_name = 'ucsd'
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
        original_dataset_name = 'trancos'
        output_dir = './saved_models_trancos/'

        train_data_config['tran1FlipResize1_trainAllValiFlip'] = train_flag.copy()
        validation_data_config['tran1Resize1_Vali'] = evaluate_flag.copy()
        test_data_config['tran1Resize1_test'] = evaluate_flag.copy()
    elif is_mall:
        original_dataset_name = 'mall'
        output_dir = './saved_models_mall/'

        train_data_config['mall1Resize05_train'] = train_flag.copy()
        validation_data_config['mall1Resize05_val'] = evaluate_flag.copy()
        # test_data_config['tran1Resize1_test'] = evaluate_flag.copy()
    elif is_ucf_qnrf:
        original_dataset_name = 'ucf_qnrf'
        output_dir = './saved_models_ucf_qnrf/'

        train_data_config['ucfQnrf9RandFlipResize1024_train'] = train_flag.copy()
        validation_data_config['ucfQnrf1Resize1024_train'] = evaluate_flag.copy()
        test_data_config['ucfQnrf1Resize1024_test'] = evaluate_flag.copy()

    # Check if there are duplicate keys in the config dict
    for key in train_data_config:
        if key in validation_data_config or key in test_data_config:
            raise Exception('duplicate dataset')
    for key in validation_data_config:
        if key in test_data_config:
            raise Exception('duplicate dataset')

    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    # load data
    # all_data = Data({**train_data_config, **validation_data_config, **test_data_config})
    # all_data = all_data.get()
    all_data = multithread_dataloader({**train_data_config, **validation_data_config, **test_data_config})

    # initialize net
    net = CrowdCount()
    network.weights_normal_init(net, dev=0.01)  # default dev=0.01
    # network.save_net('model_init.h5', net)
    if is_load_pretrained_model:
        for path in pretrained_model_path:
            network.load_net_safe(path, net)
    # network.load_net(finetune_model, net)
    # network.save_net('model_loaded.h5', net)

    net.cuda()
    net.train()

    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam([{'params': net.features.vgg16.parameters()},
                                  {'params': net.features.map.parameters()},
                                  {'params': net.features.scale.parameters()}], lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    if lr_adjust_epoch is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    standard_error_dict = dict()
    standard_error_dict['name'] = 'none'  # model name
    standard_error_dict['number'] = sys.maxsize  # number of samples
    standard_error_dict['mae'] = sys.maxsize
    standard_error_dict['mse'] = sys.maxsize
    standard_error_dict['psnr'] = sys.maxsize
    standard_error_dict['ssim'] = sys.maxsize
    standard_error_dict['game_0'] = sys.maxsize
    standard_error_dict['game_1'] = sys.maxsize
    standard_error_dict['game_2'] = sys.maxsize
    standard_error_dict['game_3'] = sys.maxsize
    best_result_dict = dict()
    for data_name in {**validation_data_config, **test_data_config}:
        best_result_dict[data_name] = standard_error_dict.copy()

    display_interval = 1000
    txt_log_info = list()
    excel_log = ExcelLog('log.xlsx')

    log_best_model_history_list = list()  # put best model name in this list after writing to the front of the log file

    for data_name in train_data_config:
        txt_log_info.append('train data: %s' % data_name)
    for data_name in validation_data_config:
        txt_log_info.append('validation data: %s' % data_name)
    for data_name in test_data_config:
        txt_log_info.append('test data: %s' % data_name)

    for epoch in range(max_epoch):
        step = -1
        train_loss = 0.0
        number_of_train_samples = 0  # number of samples which are actually used to train

        for _, param_group in enumerate(optimizer.param_groups):
            txt_log_info.append("learning rate: {:.2e}".format(float(param_group['lr'])))
        log(log_path, txt_log_info)

        if len(train_data_config) > 1:
            raise Exception('more than one train dataset is provided')

        train_data_name, = train_data_config
        train_data = all_data[train_data_name]
        data = train_data['data']

        for blob in data:
            image_data = blob['image']
            ground_truth_data = blob['density']
            roi_data = blob['roi']
            # fname = blob['filename']

            step += 1
            number_of_train_samples += 1

            estimate_map, _ = net(image_data, ground_truth=ground_truth_data, roi=roi_data)

            loss = net.loss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % display_interval == 0:
                ground_truth_map = ground_truth_data.data.cpu().numpy()
                estimate_map = estimate_map.data.cpu().numpy()
                ground_truth_count = np.sum(ground_truth_map[0])
                estimate_count = np.sum(estimate_map[0])
                if train_batch_size > 1:
                    print('epoch: %4d, step %6d, ground truth: %6.1f, estimate: %6.1f and etc' % (epoch, step, ground_truth_count, estimate_count), flush=True)
                else:
                    print('epoch: %4d, step %6d, ground truth: %6.1f, estimate: %6.1f' % (epoch, step, ground_truth_count, estimate_count), flush=True)

            if is_save_model_in_epoch and number_of_train_samples % steps_to_save_model == 0:
                model_name = '{}_{}_{}.h5'.format(original_dataset_name, epoch, step)
                save_model_path = os.path.join(output_dir, model_name)
                network.save_net(save_model_path, net)

                # evaluate the model of this epoch
                evaluate_result_dict = dict()
                for data_name in best_result_dict:
                    evaluate_data = all_data[data_name]
                    result = evaluate_model(save_model_path, evaluate_data['data'])
                    evaluate_result_dict[data_name] = result
                    txt_log_info.append('evaluate %s on %s: mae: %6.2f, mse: %6.2f, psnr: %6.2f, ssim: %6.2f, game: %6.2f, %6.2f, %6.2f, %6.2f' % (result['name'], data_name, result['mae'], result['mse'], result['psnr'], result['ssim'], result['game_0'], result['game_1'], result['game_2'], result['game_3']))

                # check if this model is new best model
                best_result_dict = compare_result(evaluate_result_dict, best_result_dict, key_error)
                for data_name in best_result_dict:
                    result = best_result_dict[data_name]
                    txt_log_info.append('best model on %s is %s with %s of %.2f' % (data_name, result['name'], key_error, result[key_error]))

                log(log_path, txt_log_info)

                excel_log.add_log(evaluate_result_dict)

        display_interval = np.ceil(number_of_train_samples / 300) * 100
        train_loss = train_loss / number_of_train_samples

        txt_log_info.append('epoch: %4d train loss: %.20f' % (epoch, train_loss))

        model_name = '{}_{}_{}.h5'.format(original_dataset_name, epoch, step)
        save_model_path = os.path.join(output_dir, model_name)
        network.save_net(save_model_path, net)

        # evaluate the model of this epoch
        evaluate_result_dict = dict()
        for data_name in best_result_dict:
            evaluate_data = all_data[data_name]
            result = evaluate_model(save_model_path, evaluate_data['data'])
            evaluate_result_dict[data_name] = result
            txt_log_info.append('evaluate %s on %s: mae: %6.2f, mse: %6.2f, psnr: %6.2f, ssim: %6.2f, game: %6.2f, %6.2f, %6.2f, %6.2f' % (result['name'], data_name, result['mae'], result['mse'], result['psnr'], result['ssim'], result['game_0'], result['game_1'], result['game_2'], result['game_3']))

        # check if this model is new best model
        best_result_dict = compare_result(evaluate_result_dict, best_result_dict, key_error)
        for data_name in best_result_dict:
            result = best_result_dict[data_name]
            txt_log_info.append('best model on %s is %s with %s of %.2f' % (data_name, result['name'], key_error, result[key_error]))

        log(log_path, txt_log_info)

        excel_log.add_log(evaluate_result_dict)

        # add info of new best model on validation data to the front of log file
        if validation_data_config:  # if validation data is not empty
            for validation_data_name in validation_data_config:
                result = best_result_dict[validation_data_name]
                best_model_on_validation_now = result['name']
                if best_model_on_validation_now not in log_best_model_history_list:
                    log_best_model_history_list.append(best_model_on_validation_now)
                    # write test result of this model to the front of the log file
                    for test_data_name in test_data_config:
                        result = evaluate_result_dict[test_data_name]
                        if result['name'] != best_model_on_validation_now:
                            raise Exception('model name on validation (%s) and test (%s) mismatch' % (best_model_on_validation_now, result['name']))
                        txt_log_info.append('best model on validation %s: evaluate %s on %s: %s: %6.2f' % (validation_data_name, result['name'], test_data_name, key_error, result[key_error]))
                        log(log_path, txt_log_info, line=len(log_best_model_history_list) - 1)

        if is_use_tensorboard:
            summary_writer.add_scalar('train loss', train_loss, epoch)
            # for data_name in train_data_config:
            #     result = evaluate_result_dict[data_name]
            #     summary_writer.add_scalar('train %s mae' % data_name, result['mae'], epoch)
            #     summary_writer.add_scalar('train %s mse' % data_name, result['mse'], epoch)
            #     summary_writer.add_scalar('train %s psnr' % data_name, result['psnr'], epoch)
            #     summary_writer.add_scalar('train %s ssim' % data_name, result['ssim'], epoch)
            #     summary_writer.add_scalar('train %s game 0' % data_name, result['game_0'], epoch)
            #     summary_writer.add_scalar('train %s game 1' % data_name, result['game_1'], epoch)
            #     summary_writer.add_scalar('train %s game 2' % data_name, result['game_2'], epoch)
            #     summary_writer.add_scalar('train %s game 3' % data_name, result['game_3'], epoch)
            for data_name in validation_data_config:
                result = evaluate_result_dict[data_name]
                summary_writer.add_scalar('validation %s mae' % data_name, result['mae'], epoch)
                summary_writer.add_scalar('validation %s mse' % data_name, result['mse'], epoch)
                summary_writer.add_scalar('validation %s psnr' % data_name, result['psnr'], epoch)
                summary_writer.add_scalar('validation %s ssim' % data_name, result['ssim'], epoch)
                summary_writer.add_scalar('validation %s game 0' % data_name, result['game_0'], epoch)
                summary_writer.add_scalar('validation %s game 1' % data_name, result['game_1'], epoch)
                summary_writer.add_scalar('validation %s game 2' % data_name, result['game_2'], epoch)
                summary_writer.add_scalar('validation %s game 3' % data_name, result['game_3'], epoch)
            for data_name in test_data_config:
                result = evaluate_result_dict[data_name]
                summary_writer.add_scalar('test %s mae' % data_name, result['mae'], epoch)
                summary_writer.add_scalar('test %s mse' % data_name, result['mse'], epoch)
                summary_writer.add_scalar('test %s psnr' % data_name, result['psnr'], epoch)
                summary_writer.add_scalar('test %s ssim' % data_name, result['ssim'], epoch)
                summary_writer.add_scalar('test %s game 0' % data_name, result['game_0'], epoch)
                summary_writer.add_scalar('test %s game 1' % data_name, result['game_1'], epoch)
                summary_writer.add_scalar('test %s game 2' % data_name, result['game_2'], epoch)
                summary_writer.add_scalar('test %s game 3' % data_name, result['game_3'], epoch)

        if lr_adjust_epoch is not None:
            scheduler.step()

    log(log_path, txt_log_info)

    if is_use_tensorboard:
        summary_writer.close()
