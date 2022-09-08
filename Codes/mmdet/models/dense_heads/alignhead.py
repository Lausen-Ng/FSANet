# -----------------------------------------------------------------------------
# Licensed under the MIT License.
# The code is based on MMDetection (https://github.com/open-mmlab/mmdetection).
# @Author  : Jixiang Wu
# @Time    : 2022/5/21 下午5:04
# -----------------------------------------------------------------------------
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d

from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

INF = 1e8


@HEADS.register_module()
class AlignHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = AlignHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 dcn_kernel=3,
                 dcn_deform_groups=4,
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_bbox_init=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.dcn_kernel = dcn_kernel
        self.dcn_padding = int((self.dcn_kernel - 1) / 2)
        self.dcn_deform_groups = dcn_deform_groups
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox_init = build_loss(loss_bbox_init)

    def _init_layers(self):
        # Initialize classification conv layers and bbox regression conv layers of the head.
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        self.regAdaption_conv = DeformConv2d(self.feat_channels, self.feat_channels,
                                             kernel_size=self.dcn_kernel, padding=self.dcn_padding,
                                             deform_groups=self.dcn_deform_groups)
        self.reg_norm = nn.GroupNorm(num_groups=32, num_channels=self.feat_channels)
        self.clsAdaption_conv = DeformConv2d(self.feat_channels, self.feat_channels,
                                             kernel_size=self.dcn_kernel, padding=self.dcn_padding,
                                             deform_groups=self.dcn_deform_groups)
        self.cls_norm = nn.GroupNorm(num_groups=32, num_channels=self.feat_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialize the offset layer of the head.
        offset_channels = self.dcn_deform_groups * self.dcn_kernel * self.dcn_kernel * 2
        self.offset_conv = nn.Conv2d(4, offset_channels, 1, bias=False)

        # Initialize predictor layers of the head.
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        self.conv_reg_init = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        # Initialize centerness layers of the head.
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """

        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        bbox_pred_init = self.conv_reg_init(reg_feat)
        bbox_pred_init_out = scale(bbox_pred_init).float()
        bbox_pred_init_out = bbox_pred_init_out.exp()

        offset = self.offset_conv(bbox_pred_init)
        cls_feat = self.relu(self.cls_norm(self.clsAdaption_conv(cls_feat, offset)))
        reg_feat = self.relu(self.reg_norm(self.regAdaption_conv(reg_feat, offset)))

        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)
        bbox_pred = scale(bbox_pred).float()

        bbox_pred = bbox_pred + bbox_pred_init_out
        centerness = self.conv_centerness(cls_feat)

        if self.training:
            return cls_score, bbox_pred, bbox_pred_init_out, centerness
        else:
            return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_pred_init_outs', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_pred_init_outs,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes, gt_labels)

        bbox_preds_convert = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds]
        center = []
        for i_lvl in range(len(bbox_preds_convert)):
            center_x = all_level_points[i_lvl][:, 0] + (
                    bbox_preds_convert[i_lvl][:, :, 2] - bbox_preds_convert[i_lvl][:, :, 0]) / 2
            center_y = all_level_points[i_lvl][:, 1] + (
                        bbox_preds_convert[i_lvl][:, :, 3] - bbox_preds_convert[i_lvl][:, :, 1]) / 2
            center.append(torch.stack([center_x, center_y], dim=-1))

        centers_list = []
        for i_img in range(num_imgs):
            single_list = []
            for i_lvl in range(len(center)):
                single_list.append(center[i_lvl][i_img, :, :])
            centers_list.append(single_list)
            
        labels_refine, bbox_targets_refine = self.get_targets_refine(centers_list, gt_bboxes, gt_labels, all_level_points)

        flatten_bbox_pred_init_outs = [
            bbox_pred_init_out.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred_init_out in bbox_pred_init_outs
        ]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        
        flatten_bbox_pred_init_outs = torch.cat(flatten_bbox_pred_init_outs)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])
        flatten_bbox_targets = torch.cat(bbox_targets)      
        flatten_labels_init = torch.cat(labels)             
        flatten_labels_refine = torch.cat(labels_refine)    
        flatten_bbox_targets_refine = torch.cat(bbox_targets_refine)    

        bg_class_ind = self.num_classes
        pos_inds_init = ((flatten_labels_init >= 0) & (flatten_labels_init < bg_class_ind)).nonzero().reshape(-1)
       
        pos_bbox_pred_init = flatten_bbox_pred_init_outs[pos_inds_init]
        pos_bbox_targets_init = flatten_bbox_targets[pos_inds_init]
        pos_centerness_targets_init = self.centerness_target_gaussian(pos_bbox_targets_init)
       
        centerness_denorm_init = max(reduce_mean(pos_centerness_targets_init.sum().detach()), 1e-6)
        pos_inds_refine = ((flatten_labels_refine >= 0) & (flatten_labels_refine < bg_class_ind)).nonzero().reshape(-1)
        num_pos_refine = torch.tensor(len(pos_inds_refine), dtype=torch.float, device=bbox_preds[0].device)
        num_pos_refine = max(reduce_mean(num_pos_refine), 1.0)
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds_refine]
        pos_bbox_targets_refine = flatten_bbox_targets_refine[pos_inds_refine]
        pos_centerness = flatten_centerness[pos_inds_refine]
        pos_centerness_targets_refine = self.centerness_target_gaussian(pos_bbox_targets_refine)
        centerness_denorm_refine = max(reduce_mean(pos_centerness_targets_refine.sum().detach()), 1e-6)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels_refine, avg_factor=num_pos_refine)

        if len(pos_inds_init) > 0:
            pos_points = flatten_points[pos_inds_init]
            pos_decoded_bbox_preds_init = self.bbox_coder.decode(pos_points, pos_bbox_pred_init)
            pos_decoded_target_preds_init = self.bbox_coder.decode(pos_points, pos_bbox_targets_init)
            loss_bbox_init = self.loss_bbox_init(
                pos_decoded_bbox_preds_init,
                pos_decoded_target_preds_init,
                weight=pos_centerness_targets_init,
                avg_factor=centerness_denorm_init)
        else:
            loss_bbox_init = pos_bbox_pred_init.sum()
        
        if len(pos_inds_refine) > 0:
            pos_points = flatten_points[pos_inds_refine]
            pos_decoded_bbox_preds_refine = self.bbox_coder.decode(pos_points, pos_bbox_preds_refine)
            pos_decoded_target_preds_refine = self.bbox_coder.decode(pos_points, pos_bbox_targets_refine)
            loss_bbox_refine = self.loss_bbox(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds_refine,
                weight=pos_centerness_targets_refine,
                avg_factor=centerness_denorm_refine)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets_refine, avg_factor=num_pos_refine)
        else:
            loss_bbox_refine = pos_bbox_preds_refine.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox_init=loss_bbox_init,
            loss_bbox_refine=loss_bbox_refine,
            loss_centerness=loss_centerness)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)   
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def get_targets_refine(self, points_list, gt_bboxes_list, gt_labels_list, points_default):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points_list (list[Tensor]): Points of each fpn image, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        single_point_list = points_list[0]
        assert len(single_point_list) == len(self.regress_ranges)
        num_levels = len(single_point_list)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            single_point_list[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                single_point_list[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points_proposals_list = [torch.cat(points, dim=0) for points in points_list]
        concat_points = torch.cat(points_default, dim=0)
        # the number of points per img, per lvl
        num_points = [expanded_regress_range.size(0) for expanded_regress_range in expanded_regress_ranges]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_refine_single,
            concat_points_proposals_list,
            gt_bboxes_list,
            gt_labels_list,
            points_default=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_refine_single(self, points_proposals, gt_bboxes, gt_labels,
                                  points_default, regress_ranges, num_points_per_lvl):
        """Compute regression and classification targets for a single image.
            Args:
                points_proposals: points所预测的回归框对应的中心点
                gt_bboxes: Ground Truth
                gt_labels: Labels
                points_default: 默认的采样points
        """
        num_points = points_proposals.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

        xs, ys = points_proposals[:, 0], points_proposals[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)  

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = ((max_regress_distance >= regress_ranges[..., 0]) & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG

        # gt bbox_targets
        xs_gt, ys_gt = points_default[:, 0], points_default[:, 1]
        xs_gt = xs_gt[:, None].expand(num_points, num_gts)
        ys_gt = ys_gt[:, None].expand(num_points, num_gts)

        left_gt = xs_gt - gt_bboxes[..., 0]
        right_gt = gt_bboxes[..., 2] - xs_gt
        top_gt = ys_gt - gt_bboxes[..., 1]
        bottom_gt = gt_bboxes[..., 3] - ys_gt
        bbox_targets_gt = torch.stack((left_gt, top_gt, right_gt, bottom_gt), -1)

        bbox_targets_gt = bbox_targets_gt[range(num_points), min_area_inds]

        return labels, bbox_targets_gt

    def centerness_target_gaussian(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

            Returns:
                Tensor: Centerness target.
        """
        # (left, top, right, bottom)
        left    = pos_bbox_targets[:, 0]
        top     = pos_bbox_targets[:, 1]
        right   = pos_bbox_targets[:, 2]
        bottom  = pos_bbox_targets[:, 3]

        width = left + right
        height = top + bottom

        if len(left) == 0:
            centerness_targets = pos_bbox_targets[:, 0]
        else:
            delta_x = torch.abs(left - right) / 2
            delta_y = torch.abs(top - bottom) / 2
            centerness_targets = torch.exp(- 4 * (delta_x / width) ** 2 - 4 * (delta_y / height) ** 2)

        return centerness_targets

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map size.

        This function will be deprecated soon.
        """
        warnings.warn(
            '`_get_points_single` in `FCOSHead` will be '
            'deprecated soon, we support a multi level point generator now'
            'you can get points of a single level feature map '
            'with `self.prior_generator.single_level_grid_priors` ')

        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points
