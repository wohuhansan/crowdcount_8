import torch
import torch.nn as nn
# import numpy as np
import torch.nn.functional as functional
from functools import reduce

# from src import network
from src.models import Model


class CrowdCount(nn.Module):
    def __init__(self):
        super(CrowdCount, self).__init__()
        self.features = Model()
        self.my_loss = None

    @property
    def loss(self):
        return self.my_loss

    def forward(self, im_data, ground_truth=None, roi=None):
        im_data = im_data.cuda()
        if roi is not None:
            roi = roi.cuda()
        estimate_map, foreground_mask, visual_dict = self.features(im_data, roi)

        if self.training:
            ground_truth_map = ground_truth.cuda()
            self.my_loss = self.build_loss(ground_truth_map, estimate_map, foreground_mask)

        return estimate_map, visual_dict

    def build_loss(self, ground_truth_map, estimate_map, foreground_mask):
        if ground_truth_map.shape != estimate_map.shape:
            raise Exception('ground truth shape and estimate shape mismatch')

        ground_truth_map = ground_truth_map * foreground_mask
        estimate_map = estimate_map * foreground_mask

        density_loss = torch.mean((ground_truth_map - estimate_map) ** 2)
        return density_loss

        # pooling_loss = self.pooling_loss(ground_truth_map, estimate_map, 1)
        # return pooling_loss

    @staticmethod
    def build_block(x, size):
        # x shape=(1, c, h, w)
        from math import ceil
        height = x.shape[2]
        width = x.shape[3]
        padding_height = ceil((ceil(height / size) * size - height) / 2)
        padding_width = ceil((ceil(width / size) * size - width) / 2)
        return functional.avg_pool2d(x, size, stride=size, padding=(padding_height, padding_width), count_include_pad=True) * size * size

    def pooling_loss(self, ground_truth, estimate, block_size=4):
        square_error = (ground_truth - estimate) ** 2
        element_amount = reduce(lambda x, y: x * y, square_error.shape)
        block_square_error = self.build_block(square_error / element_amount, block_size)
        block_ground_truth = self.build_block(ground_truth, block_size)
        block_loss = torch.sum(block_square_error / (block_ground_truth + 1))
        return block_loss

    def rank_loss(self, ground_truth_map, estimate_map, block_size=None, output_size=None):
        if block_size is not None and output_size is not None:
            raise Exception('both block_size and output_size are provided')
        if block_size is not None:
            block_ground_truth = self.build_block(ground_truth_map, block_size)
            block_estimate = self.build_block(estimate_map, block_size)
        elif output_size is not None:
            block_ground_truth = functional.adaptive_avg_pool2d(ground_truth_map, output_size)
            block_estimate = functional.adaptive_avg_pool2d(estimate_map, output_size)
        else:
            raise Exception('none of block_size or output_size is provided')
        ground_truth_vector = torch.reshape(block_ground_truth, (-1,))
        estimate_vector = torch.reshape(block_estimate, (-1,))
        _, ground_truth_indices = torch.sort(ground_truth_vector)
        _, estimate_indices = torch.sort(estimate_vector)
        rank_loss = torch.mean(torch.abs(ground_truth_indices.to(torch.float32) - estimate_indices.to(torch.float32))) / ground_truth_indices.shape[0]
        return rank_loss
