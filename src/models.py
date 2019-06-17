import torch
import torch.nn as nn
import torch.nn.functional as functional
import cv2
import numpy as np

from src.network import Conv2d, ConvTranspose2d
from src.utils import ndarray_to_tensor


class Model(nn.Module):
    def __init__(self, bn=False):
        super(Model, self).__init__()

        self.prior = nn.Sequential(Conv2d(3, 64, 3, same_padding=True),
                                   Conv2d(64, 64, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(64, 128, 3, same_padding=True),
                                   Conv2d(128, 128, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(128, 256, 3, same_padding=True),
                                   Conv2d(256, 256, 3, same_padding=True),
                                   Conv2d(256, 256, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(256, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 256, 1, same_padding=True),
                                   ConvTranspose2d(256, 128, 2, stride=2, padding=0),
                                   Conv2d(128, 128, 3, same_padding=True),
                                   Conv2d(128, 3, 1, same_padding=True))

        self.vgg16 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True),
                                   Conv2d(64, 64, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(64, 128, 3, same_padding=True),
                                   Conv2d(128, 128, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(128, 256, 3, same_padding=True),
                                   Conv2d(256, 256, 3, same_padding=True),
                                   Conv2d(256, 256, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(256, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   nn.MaxPool2d(2),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 512, 3, same_padding=True),
                                   Conv2d(512, 256, 1, same_padding=True),
                                   ConvTranspose2d(256, 128, 2, stride=2, padding=0),
                                   Conv2d(128, 128, 3, same_padding=True, dilation=2),
                                   Conv2d(128, 128, 3, same_padding=True, dilation=2),
                                   Conv2d(128, 128, 3, same_padding=True, dilation=2))

        self.map = nn.Sequential(Conv2d(128, 128, 3, same_padding=True),
                                 Conv2d(128, 2, 1, same_padding=True))

        self.scale = nn.Sequential(Conv2d(128, 128, 3, same_padding=True),
                                   Conv2d(128, 2, 1, same_padding=True, relu=False),
                                   nn.AdaptiveAvgPool2d(1),
                                   nn.Hardtanh(-1.0, 1.0))

    def forward(self, im_data, roi=None):
        with torch.no_grad():
            x_prior = self.prior(im_data)
            flag = torch.argmax(x_prior, dim=1, keepdim=True)

            background_mask = (flag == 0).to(torch.float32)
            foreground_mask = 1 - background_mask
            resized_foreground_mask = functional.interpolate(1 - background_mask, scale_factor=8.0, mode='nearest')

            # mask = None
            # for i in range(1, x_prior.shape[1]):
            #     if mask is None:
            #         mask = (flag == i).to(torch.float32)
            #     else:
            #         mask = torch.cat((mask, (flag == i).to(torch.float32)), dim=1)

            low_density_mask = (flag == 1).to(torch.float32)
            high_density_mask = (flag == 2).to(torch.float32)
            low_density_mask_np = low_density_mask.data.cpu().numpy()
            high_density_mask_np = high_density_mask.data.cpu().numpy()
            dilate_kernel = np.ones((2, 2))
            dilated_low_density_mask_list = list()
            dilated_high_density_mask_list = list()
            for batch in range(low_density_mask_np.shape[0]):
                this_mask = cv2.dilate(low_density_mask_np[batch][0], dilate_kernel, iterations=1)
                dilated_low_density_mask_list.append(ndarray_to_tensor(this_mask.reshape((1, 1, this_mask.shape[0], this_mask.shape[1]))).cuda())
                this_mask = cv2.dilate(high_density_mask_np[batch][0], dilate_kernel, iterations=1)
                dilated_high_density_mask_list.append(ndarray_to_tensor(this_mask.reshape((1, 1, this_mask.shape[0], this_mask.shape[1]))).cuda())
            dilated_low_density_mask = torch.cat(dilated_low_density_mask_list, dim=0)
            dilated_high_density_mask = torch.cat(dilated_high_density_mask_list, dim=0)

            overlap_mask = (low_density_mask * dilated_high_density_mask + high_density_mask * dilated_low_density_mask) * foreground_mask
            new_low_density_mask = low_density_mask * (1 - overlap_mask) * foreground_mask
            new_high_density_mask = high_density_mask * (1 - overlap_mask) * foreground_mask

        x1 = self.vgg16(im_data * resized_foreground_mask)
        maps = self.map(x1)
        scales = self.scale(x1) + 1

        low_density_map, high_density_map = torch.chunk(maps, 2, dim=1)
        low_density_scale, high_density_scale = torch.chunk(scales, 2, dim=1)

        scaled_low_density_map = low_density_map * low_density_scale * new_low_density_mask
        scaled_high_density_map = high_density_map * high_density_scale * new_high_density_mask
        scaled_overlap_map = (low_density_map * low_density_scale * overlap_mask + high_density_map * high_density_scale * overlap_mask) / 2
        scaled_maps = torch.cat((scaled_low_density_map, scaled_high_density_map, scaled_overlap_map), dim=1)
        density_map = torch.sum(scaled_maps, 1, keepdim=True)

        if roi is not None:
            density_map = density_map * roi

        visual_dict = dict()
        # visual_dict['score'] = x_score_maps
        # visual_dict['class'] = x_class_maps
        visual_dict['density'] = density_map
        visual_dict['raw_maps'] = maps
        visual_dict['scaled_maps'] = scaled_maps
        visual_dict['masks'] = torch.cat((background_mask, new_low_density_mask, new_high_density_mask, overlap_mask), dim=1)

        return density_map, foreground_mask, visual_dict
