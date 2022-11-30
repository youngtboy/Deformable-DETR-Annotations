# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape  # [B, num_points_all_level, num_head, head_dim]
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape  # [B, num_points_all_level, n_heads, n_levels, n_points, 2]

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1) # 将每个level的value分开 List[Tensor] 每个元素shape [B, num_point_per_level, n_head, head_dim]

    sampling_grids = 2 * sampling_locations - 1  # 坐标 [0,1]->[-1,1] 为了使用F.grid_sample


    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):  # 遍历每个level

        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # value_list[lid_] -> [B, num_point_per_level, n_head, head_dim] -> [B, num_point_per_level, n_head*head_dim] -> 
        # [B, n_head*head_dim, num_point_per_level] ->  [B*n_head, head_dim, H, W]
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)


        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        #  sampling_grids[:, :, :, lid_] -> [B, num_points_all_level, n_heads, n_points, 2] -> 
        # [B, n_heads, num_points_all_level, n_points, 2] ->  [B*n_heads, num_points_all_level, n_points, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)  # 所有位置在当前的特征层上的采样点


        # N_*M_, D_, Lq_, P_
        # sampling_value_l_： [B*n_head, head_dim, num_points_all_level, n_points] # 这里的4代表4个点
        # 每个head为每个point在该特征层均采样了4个点，即所有位置在当前特征层上采样得到的结果
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)


        sampling_value_list.append(sampling_value_l_)


    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # [B, num_points_all_level,  n_heads, n_levels*n_points] ->  [B,n_heads,  num_points_all_level, n_levels*n_points]
    # -> [B*n_heads, 1, num_points_all_level, n_levels*n_points]
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)


    # torch.stack(sampling_value_list, dim=-2).flatten(-2) 
    # [B*8, head_dim, num_points_all_level, n_level, n_points]  这里第一个4的维度代表4个level 第二个4的维度代表4个点 -> [B*n_head, head_dim, num_points_all_level, n_level*n_points]
    # [B*8, head_dim, num_points_all_level, n_level*n_points]*[B*n_head, 1, num_points_all_level, n_level*n_points] = [B*n_head, head_dim, num_points_all_level, n_level*n_points]
    # -> [B*n_head, head_dim, num_points_all_level] -> [B, n_head*head_dim, num_points_all_levels ]
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous() # [B,num_points_all_levels, 256]
