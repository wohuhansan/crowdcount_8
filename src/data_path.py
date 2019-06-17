from os.path import join


class DataPath:
    def __init__(self):
        # base_path = r'D:/PycharmProjects'
        base_path = r'/media/d/PycharmProjects'
        self.data_path = dict()

        path = dict()
        path['image'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_64_64/train')
        path['gt'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_64_64/train_den')
        path['roi'] = None
        self.data_path['shtA0RandomOverturn_64_64_train'] = path

        path = dict()
        path['image'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_128_256/train')
        path['gt'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_128_256/train_den')
        path['roi'] = None
        self.data_path['shtA0RandomOverturn_128_256_train'] = path

        path = dict()
        path['image'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_128_256_more_than_one_pedestrain/train')
        path['gt'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_overturn_rgb_128_256_more_than_one_pedestrain/train_den')
        path['roi'] = None
        self.data_path['shtA0RandomOverturn_128_256_more1_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_flip_rgb_128_256_more_than_ten_pedestrian/train')
        path['gt'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_0_random_flip_rgb_128_256_more_than_ten_pedestrian/train_den')
        path['roi'] = None
        self.data_path['shtA0RandFlip_128_256_more10_train'] = path

        path = dict()
        path['image'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/test')
        path['gt'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/test_den')
        path['roi'] = None
        self.data_path['shtA1Resize1_test'] = path

        path = dict()
        path['image'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/train')
        path['gt'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb/train_den')
        path['roi'] = None
        self.data_path['shtA1Resize1_train'] = path

        path = dict()
        path['image'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb_times32/test')
        path['gt'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_resize_1_rgb_times32/test_den')
        path['roi'] = None
        self.data_path['shtA1Resize1Times32_test'] = path

        path = dict()
        path['image'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/train')
        path['gt'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/train_den')
        path['roi'] = None
        self.data_path['shtA1_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/test')
        path['gt'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_1_rgb/test_den')
        path['roi'] = None
        self.data_path['shtA1_test'] = path

        path = dict()
        path['image'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train')
        path['gt'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train_den')
        path['roi'] = None
        self.data_path['shtA9RandomOverturn_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train_without_validation')
        path['gt'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/train_without_validation_den')
        path['roi'] = None
        self.data_path['shtA9RandFlip_trainNoVali'] = path

        path = dict()
        path['image'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/validation')
        path['gt'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb/validation_den')
        path['roi'] = None
        self.data_path['shtA9RandFlip_vali'] = path

        path = dict()
        path['image'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb_times32/train')
        path['gt'] = join(base_path, 'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_A_patches_9_random_overturn_rgb_times32/train_den')
        path['roi'] = None
        self.data_path['shtA9RandomOverturnTimes32_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_B_patches_1_rgb/train')
        path['gt'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_B_patches_1_rgb/train_den')
        path['roi'] = None
        self.data_path['shtB1_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_B_patches_1_rgb/test')
        path['gt'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_4/shanghaitech_part_B_patches_1_rgb/test_den')
        path['roi'] = None
        self.data_path['shtB1_test'] = path

        path = dict()
        path['image'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_9_random_overturn_rgb_more_than_one_pedestrain/train')
        path['gt'] = join(base_path, r'data/shanghaitech/formatted_trainval_15_15/shanghaitech_part_B_patches_9_random_overturn_rgb_more_than_one_pedestrain/train_den')
        path['roi'] = None
        self.data_path['shtB9RandFlip_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/1/train')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/1/train_den')
        path['roi'] = None
        self.data_path['ucf9RandFlip_train1'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/1/val')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/1/val_den')
        path['roi'] = None
        self.data_path['ucf1_test1'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/2/train')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/2/train_den')
        path['roi'] = None
        self.data_path['ucf9RandFlip_train2'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/2/val')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/2/val_den')
        path['roi'] = None
        self.data_path['ucf1_test2'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/3/train')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/3/train_den')
        path['roi'] = None
        self.data_path['ucf9RandFlip_train3'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/3/val')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/3/val_den')
        path['roi'] = None
        self.data_path['ucf1_test3'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/4/train')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/4/train_den')
        path['roi'] = None
        self.data_path['ucf9RandFlip_train4'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/4/val')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/4/val_den')
        path['roi'] = None
        self.data_path['ucf1_test4'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/5/train')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_9_random_rgb_overturn/5/train_den')
        path['roi'] = None
        self.data_path['ucf9RandFlip_train5'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/5/val')
        path['gt'] = join(base_path, r'data/ucf_cc_50/formatted_trainval_15_4/ucf_cc_50_patches_1_rgb/5/val_den')
        path['roi'] = None
        self.data_path['ucf1_test5'] = path

        path = dict()
        path['image'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train')
        path['gt'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train_den')
        path['roi'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb_overturn/train_roi')
        self.data_path['we1Flip_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/1')
        path['gt'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/1')
        path['roi'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all')
        self.data_path['we1_test1'] = path

        path = dict()
        path['image'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/2')
        path['gt'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/2')
        path['roi'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all')
        self.data_path['we1_test2'] = path

        path = dict()
        path['image'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/3')
        path['gt'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/3')
        path['roi'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all')
        self.data_path['we1_test3'] = path

        path = dict()
        path['image'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/4')
        path['gt'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/4')
        path['roi'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all')
        self.data_path['we1_test4'] = path

        path = dict()
        path['image'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test/5')
        path['gt'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_den/5')
        path['roi'] = join(base_path, r'data/WorldExpo10/formatted_trainval_15_4/worldexpo_patches_1_rgb/test_roi/all')
        self.data_path['we1_test5'] = path

        path = dict()
        path['image'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn')
        path['gt'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn_den')
        path['roi'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_overturn_resize_1_rgb/train_all_val_overturn_roi')
        self.data_path['tran1FlipResize1_trainAllValiFlip'] = path

        path = dict()
        path['image'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val')
        path['gt'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val_den')
        path['roi'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/val_roi')
        self.data_path['tran1Resize1_Vali'] = path

        path = dict()
        path['image'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test')
        path['gt'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test_den')
        path['roi'] = join(base_path, r'data/trancos/formatted_trainval_15_10/trancos_patches_1_resize_1_rgb/test_roi')
        self.data_path['tran1Resize1_test'] = path

        path = dict()
        path['image'] = join(base_path, r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train')
        path['gt'] = join(base_path, r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train_den')
        path['roi'] = join(base_path, r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/train_roi')
        self.data_path['mall1Resize05_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val')
        path['gt'] = join(base_path, r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val_den')
        path['roi'] = join(base_path, r'data/mall/formatted_trainval_15_4/mall_patches_1_resize_05_rgb/val_roi')
        self.data_path['mall1Resize05_val'] = path

        path = dict()
        path['image'] = join(base_path, r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train')
        path['gt'] = join(base_path, r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_den')
        path['roi'] = join(base_path, r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/train_roi')
        self.data_path['air1_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test')
        path['gt'] = join(base_path, r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_den')
        path['roi'] = join(base_path, r'data/airport/formatted_trainval_15_4/airport_patches_1_rgb/test_roi')
        self.data_path['air1_test'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_0_random_flip_rgb_128_256_more1_resize1024/train')
        path['gt'] = join(base_path, r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_0_random_flip_rgb_128_256_more1_resize1024/train_den')
        path['roi'] = None
        self.data_path['ucfQnrf0RandFlip_128_256_more1Resize1024_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_1_rgb_resize1024/train')
        path['gt'] = join(base_path, r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_1_rgb_resize1024/train_den')
        path['roi'] = None
        self.data_path['ucfQnrf1Resize1024_train'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_1_rgb_resize1024/test')
        path['gt'] = join(base_path, r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_1_rgb_resize1024/test_den')
        path['roi'] = None
        self.data_path['ucfQnrf1Resize1024_test'] = path

        path = dict()
        path['image'] = join(base_path, r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_9_random_flip_rgb_resize1024/train')
        path['gt'] = join(base_path, r'data/ucf_qnrf/kernel_15_4/ucf_qnrf_patches_9_random_flip_rgb_resize1024/train_den')
        path['roi'] = None
        self.data_path['ucfQnrf9RandFlipResize1024_train'] = path

    def get_path(self, name):
        return self.data_path[name]
