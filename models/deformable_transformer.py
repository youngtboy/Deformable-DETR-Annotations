# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t # 每个数都被编成128的向量
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos  # [B, 300, 512]

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape  # [B, num_points_all_levels, 256]
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes): # 遍历每个特征层,单独处理  为每个特征点生成proposal （x,y, w, h）
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)  # [B, H, W, 1]
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)  # [B,]
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)  # [B,]

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))  # [H, W]
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # [H, W, 2]

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)  # [B, 1, 1, 2]
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale  # [B, H, W, 2]/[B, 1, 1, 2]=[B, H, W, 2]
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)  # base object scale = 0.05
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)  # [B, H*W, 4]
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)  # [B, all_point_all_level, 4]
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))  # [B, all_point_all_level, 4]

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))   # [B, all_point_all_level, 256]
        return output_memory, output_proposals  # [B, all_point_all_level, 256]  [B, all_point_all_level, 4]

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)  # 记录每个特征层的尺寸
            src = src.flatten(2).transpose(1, 2)  # [B, 256, H, W] -> [B, HW, 256]
            mask = mask.flatten(1)  # [B, H, W] -> [B, HW]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [B, 256, H, W] -> [B, HW, 256]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # 为每个level的位置编码加可学习层级编码 [B, HW, 256]+[1, 1, 256]=[B, HW, 256]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # 将所有level的结果拼接在一起 [B, num_points_all_level, 256]
        mask_flatten = torch.cat(mask_flatten, 1)  # [B, num_points_all_level]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # [B, num_points_all_level, 256]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # [num_level, 2]

        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))  # 记录展平后每个level的起始index
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # 根据mask计算每个level的 valid ratio [B, num_level, 2]
        
        # 至此得到六个重要的量src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios
        # src_flatten:Tensor 所有level展平的特征拼接在一起的结果 Tensor shape [B, num_points_all_level, 256]
        # mask_flatten:Tensor 所有level展平的mask拼接在一起的结果 Tensor shape [B, num_points_all_level]
        # lvl_pos_embed_flatten:Tensor 所有level展平的位置编码拼接在一起的结果 Tensor shape [B, num_points_all_level, 256]
        # spatial_shapes:Tensor 所有level的特征尺寸  Tensor shape [num_level, 2]
        # level_start_index:Tensor 展平后每个level的起始index  Tensor shape [num_level]
        # valid_ratios:Tensor 所有level的 valid ratio Tensor shape [B, num_level, 2]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # memory:Tensor shape [B, num_points_all_level, 256]


        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:  # DINO对这里做了改进
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)  # [B, num_point_all_level, 91]
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals  # [B, num_point_all_level, 4]

            topk = self.two_stage_num_proposals  # 300
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]  # [B. 300]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()  # [B, 300, 4]
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))  # [B, 300, 512]
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)  #可学习参数 query_embed Tensor shape [300, 256] tgt Tensor shape [300, 256]
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)  # [B, 300, 256]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)  # [B, 300, 256]
            reference_points = self.reference_points(query_embed).sigmoid()  # decoder时使用的参考点由query_embed预测得到 [B, 300, 2]
            init_reference_out = reference_points # [B, 30000, 2]

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references  # [B, num_queries, 2]
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """ valid_ratios [B, 4, 2] 4这个维度代表有4个特征层"""
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):  # 遍历每个特征层

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),  # 生成该特征层空间坐标
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))  # ref_x res_y [H, W]
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)  # y坐标归一化  [B, HW]
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)  # x坐标归一化  [B, HW]
            ref = torch.stack((ref_x, ref_y), -1)  # stack x, y坐标  [B, HW, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # 拼接所有特征层的坐标 [B, num_points_all_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # [B,num_points_all_level, 1, 2 ]*[B, 1, 4, 2] = [B, num_points_all_level, 4, 2] 每个point位置对应在每个level上的位置信息
        return reference_points  # [B, num_points_all_level, 4, 2] reference_points一定都在有效区域内

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)  
        # Tensor shape[B, num_points_all_levle, num_level, 4] encoder的每个layer使用的是相同的reference_points
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)  # 这里的q和k是一样的
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points, # [B， 300， 4，2]
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:  # Iterative Bounding Box Refinement模式下会使用
                # [B, 300, 1, 4] * [B, 1, 4, 4] = [B, 300, 4, 4] 第一个4为level维度，第二个4为(x, y, w, h)
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                # [B, 300, 1,  2] * [B, 1, 4, 2]  = [B, 300, 4, 2]
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]  #为decoder生成的参考点

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)  # 通过3层Linear得到offset 且每个decoder layer的3层Linear参数不共享  [B, 300, 4]
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points) # 修正参考点 [B, 300, 4] (x, y, w, h)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)  # 修正参考点  [B, 300, 4]
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()  # DINO里提出对了这里的改进

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)
