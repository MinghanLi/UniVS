# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

import torchvision.transforms as T

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from .point_features import get_uncertain_point_coords_on_grid_boxvis, get_uncertain_point_coords_inbox
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils.box_ops import matched_boxlist_giou


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the number of masks

    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_with_weight_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the number of masks
        weights: weights
    Returns:
        Loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    weights = weights.flatten(1)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    return loss.sum() / max(num_masks, 0.5)


sigmoid_ce_with_weight_loss_jit = torch.jit.script(
    sigmoid_ce_with_weight_loss
)  # type: torch.jit.ScriptModule


def dice_coefficient_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = (inputs**2).sum(-1) + (targets**2).sum(-1)
    loss = 1 - numerator / denominator.clamp(min=1e-3)
    return loss.sum() / num_masks


dice_coefficient_loss_jit = torch.jit.script(
    dice_coefficient_loss
)  # type: torch.jit.ScriptModule


def pairwise_loss(
        mask: torch.Tensor,
        tgt_box_mask: torch.Tensor,
        img_lab: torch.Tensor,
        batch_indices: torch.Tensor,
        color_thresh: float,
        num_masks: float,
        stride: int
):
    """
    Args:
        mask: predicted masks after sigmoid activation (N, H, W) or (N, T, H, W)
        tgt_box_mask: ground-truth boxed masks (N, H, W) or (N, T, H, W)
        img_lab: original image in LAB color space (B, 3, H, W) or (B, T, 3, H, W), B denotes batch size
        batch_indices:
        color_thresh: pairwise color thresh, default=0.3
        num_masks: the averaged number of masks
        stride: the stride of pairwise consistency

    Returns:

    """
    # Cond1: the central pixel should belong to foreground
    is_inbox_h = (tgt_box_mask[..., stride:, :] | tgt_box_mask[..., :-stride, :]).flatten(1)  # BxPh
    is_inbox_w = (tgt_box_mask[..., :, stride:] | tgt_box_mask[..., :, :-stride]).flatten(1)  # BxPw

    # Cond2: the paired pixels should have similar color in lab space
    img_tv_h = torch.exp(-torch.norm(img_lab[..., stride:, :] - img_lab[..., :-stride, :], dim=-3) * 0.5).flatten(1)
    img_tv_w = torch.exp(-torch.norm(img_lab[..., :, stride:] - img_lab[..., :, :-stride], dim=-3) * 0.5).flatten(1)
    valid_h = (img_tv_h >= color_thresh)[batch_indices] & is_inbox_h  # NxPh
    valid_w = (img_tv_w >= color_thresh)[batch_indices] & is_inbox_w  # NxPw

    loss_type = 'pairwise'
    if loss_type == 'tv':
        # pairwise (keep smooth)
        mask_tv_h = torch.pow(mask[..., stride:, :] - mask[..., :-stride, :], 2).flatten(1)  # NxPh, where Ph=T*(H-1)*W
        mask_tv_w = torch.pow(mask[..., :, stride:] - mask[..., :, :-stride], 2).flatten(1)  # NxPw, where Pw=T*H*(W-1)

        num_valid = valid_h.sum(1) + valid_w.sum(1)
        pre_loss = (mask_tv_h * valid_h).sum(1) + (mask_tv_w * valid_w).sum(1)
        loss = pre_loss / num_valid.clamp(min=1)  # N

        loss = loss.sum() / num_masks
    else:
        # pairwise loss (keep boundary)
        log_fg_prob = F.logsigmoid(mask)
        log_bg_prob = F.logsigmoid(-mask)

        # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
        # we compute the probability in log space to avoid numerical instability
        log_same_fg_prob_h = (log_fg_prob[..., stride:, :] + log_fg_prob[..., :-stride, :]).flatten(1)
        log_same_bg_prob_h = (log_bg_prob[..., stride:, :] + log_bg_prob[..., :-stride, :]).flatten(1)
        log_same_fg_prob_w = (log_fg_prob[..., :, stride:] + log_fg_prob[..., :, :-stride]).flatten(1)
        log_same_bg_prob_w = (log_bg_prob[..., :, stride:] + log_bg_prob[..., :, :-stride]).flatten(1)

        max_h = torch.max(log_same_fg_prob_h, log_same_bg_prob_h)
        log_same_prob_h = torch.log(
            torch.exp(log_same_fg_prob_h - max_h) +
            torch.exp(log_same_bg_prob_h - max_h)
        ) + max_h

        max_w = torch.max(log_same_fg_prob_w, log_same_bg_prob_w)
        log_same_prob_w = torch.log(
            torch.exp(log_same_fg_prob_w - max_w) +
            torch.exp(log_same_bg_prob_w - max_w)
        ) + max_w

        loss_h = (-log_same_prob_h * valid_h).sum(dim=-1) / valid_h.sum(dim=-1).clamp(min=1)
        loss_w = (-log_same_prob_w * valid_w).sum(dim=-1) / valid_w.sum(dim=-1).clamp(min=1)
        loss = 0.5 * (loss_h + loss_w).sum() / num_masks

    return loss


pairwise_loss_jit = torch.jit.script(
    pairwise_loss
)  # type: torch.jit.ScriptModule


def get_bounding_boxes(masks):
    """
    Args:
        masks: NxHxW or BxNxHxW
    Returns:
        boxes: Nx4 or BxNx4
        If a mask is empty, it's bounding box will be all zero.
    """
    H, W = masks.shape[-2:]
    x_any = torch.any(masks, dim=1)
    x_any_cumsum = x_any.cumsum(dim=-1)
    x_any_cumsum[x_any_cumsum != 1] = W
    x_min = x_any_cumsum.argmin(dim=-1)

    x_any_cumsum = x_any[:, range(W)[::-1]].cumsum(dim=-1)
    x_any_cumsum[x_any_cumsum != 1] = W
    x_max_val, x_max = x_any_cumsum.min(dim=-1)
    x_max = (x_max_val == 1) * (W - x_max)

    y_any = torch.any(masks, dim=2)
    y_any_cumsum = y_any.cumsum(dim=-1)
    y_any_cumsum[y_any_cumsum != 1] = H
    y_min = y_any_cumsum.argmin(dim=-1)

    y_any_cumsum = y_any[:, range(H)[::-1]].cumsum(dim=-1)
    y_any_cumsum[y_any_cumsum != 1] = H
    y_max_val, y_max = y_any_cumsum.min(dim=-1)
    y_max = (y_max_val == 1) * (H - y_max)

    boxes = torch.stack([x_min / float(W), y_min / float(H),
                         x_max / float(W), y_max / float(H)], dim=-1)

    return boxes


def calculate_uncertainty(logits):
    """
    We estimate uncertainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def calculate_uncertainty_pseudo(logits):
    """
    We estimate uncertainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class TeacherSetPseudoMask(nn.Module):
    def __init__(self, matcher):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
        """
        super().__init__()
        self.matcher = matcher

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc.
                      The mask in targets is generated mask via bounding boxes
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, cost_type='teacher')

        src_masks = outputs_without_aux["pred_masks"].clone().detach()  # B, cQ, Hp, Wp
        src_logits = F.softmax(outputs_without_aux["pred_logits"], dim=-1).clone().detach()  # B, cQ, k
        for i, ((src_idx, tgt_idx), target) in enumerate(zip(indices, targets)):
            assert len(tgt_idx) == target["masks"].shape[0]
            tgt_h, tgt_w = target["masks"].shape[-2:]

            tgt_idx, tgt_idx_sorted = tgt_idx.sort()
            src_idx = src_idx[tgt_idx_sorted]

            tgt_mask_pseudo_soft = src_masks[i, src_idx].sigmoid()
            tgt_mask_pseudo_hard = tgt_mask_pseudo_soft > 0.5

            tgt_labels = target["labels"][tgt_idx]
            src_scores = src_logits[i, src_idx, tgt_labels]

            numerator = (tgt_mask_pseudo_soft.flatten(1) * tgt_mask_pseudo_hard.flatten(1)).sum(1)
            denominator = tgt_mask_pseudo_hard.flatten(1).sum(1)
            mask_scores = numerator / (denominator + 1e-6)

            tgt_mask_box = target["masks"][tgt_idx]
            tgt_mask_pseudo_up = F.interpolate(tgt_mask_pseudo_soft.unsqueeze(1), (tgt_h, tgt_w),
                                               mode='bilinear', align_corners=False).gt(0.5).squeeze(1)  # cQ, Ht, Wt

            target["mask_pseudo_scores"] = src_scores * mask_scores
            target["masks_pseudo"] = tgt_mask_box * tgt_mask_pseudo_up

        return targets


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, vita_last_layer_num,
                 boxvis_enabled, boxvis_pairwise_size, boxvis_pairwise_dilation,
                 boxvis_pairwise_color_thresh, boxvis_pairwise_warmup_iters, boxvis_boxdet_on,
                 boxvis_ema_enabled,
                 ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.vita_last_layer_num = vita_last_layer_num

        # box-supervised video instance segmentation
        self.boxvis_enabled = boxvis_enabled
        self.boxvis_pairwise_size = boxvis_pairwise_size
        self.boxvis_pairwise_dilation = boxvis_pairwise_dilation
        self.boxvis_pairwise_color_thresh = boxvis_pairwise_color_thresh
        self.boxvis_pairwise_warmup_iters = boxvis_pairwise_warmup_iters
        self.boxvis_boxdet_on = boxvis_boxdet_on
        self.boxvis_ema_enabled = boxvis_ema_enabled
        self.register_buffer("_iter", torch.zeros([1]))

        assert self.boxvis_pairwise_size % 2 == 1
        dsize = torch.arange(self.boxvis_pairwise_size) - self.boxvis_pairwise_size//2
        dsize = dsize * self.boxvis_pairwise_dilation
        dwh = torch.stack([dsize.reshape(1, -1).repeat(self.boxvis_pairwise_size, 1),
                           dsize.reshape(-1, 1).repeat(1, self.boxvis_pairwise_size)],
                          dim=-1).reshape(-1, 2)
        dwh = torch.cat([dwh[self.boxvis_pairwise_size**2//2:], dwh[:self.boxvis_pairwise_size**2//2]])
        self.register_buffer("boxvis_pairwise_dwh", dwh)

        kernel_size = 3
        assert kernel_size % 2 == 1 and kernel_size > 1
        noise_reduced_kernel = torch.ones(kernel_size, kernel_size)
        noise_reduced_kernel[kernel_size//2, kernel_size//2] = noise_reduced_kernel.sum() - 1
        self.noise_reduced_kernel = noise_reduced_kernel.reshape(1, 1, kernel_size, kernel_size) / noise_reduced_kernel.sum()
        self.noise_reduced_kernel_size = kernel_size

        self.mask_pseudo_score_thresh = 0.2

    def loss_labels(self, outputs, targets, indices, num_masks, last_layer):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, last_layer):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords: N x 12544 x 2
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_with_box_supervised(self, outputs, targets, indices, num_masks, last_layer):
        src_idx = self._get_src_permutation_idx(indices)
        batch_i, tgt_i = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]

        src_masks = src_masks[src_idx]  # NxHpxWp
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)  # NxHxW
        src_h, src_w = src_masks.shape[-2:]
        tgt_h, tgt_w = target_masks.shape[-2:]

        # No need to align the size between predicted masks and ground-truth masks
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        # image_lab_color: 3xHxW, image_lab_color_masks: HxW
        image_lab_color = torch.cat([t["image_lab_color"][None] for t in targets]).to(src_masks)  # Bx3xHxW

        loss_type = 'downsample'
        # the predicted masks are inaccurate in the initial iterations
        if loss_type == 'upsample':
            pair_stride = 2
            src_masks = F.interpolate(src_masks, (tgt_h, tgt_w),
                                      mode='bilinear', align_corners=False)
        else:
            pair_stride = 1
            with torch.no_grad():
                target_masks = F.interpolate(target_masks, (src_h, src_w),
                                             mode='bilinear', align_corners=False).gt(0.5)
                image_lab_color = F.interpolate(image_lab_color, (src_h, src_w),
                                                mode='bilinear', align_corners=False)  # Bx3xHxW

        # ------------------ points out of bounding box / project term --------------------------
        # max simply selects the greatest value to backprop, so max is the identity operation for that one element
        mask_losses_y = dice_coefficient_loss_jit(
            src_masks.sigmoid().max(dim=-2, keepdim=True)[0],
            target_masks.max(dim=-2, keepdim=True)[0],
            num_masks
        )
        mask_losses_x = dice_coefficient_loss_jit(
            src_masks.sigmoid().max(dim=-1, keepdim=True)[0],
            target_masks.max(dim=-1, keepdim=True)[0],
            num_masks
        )
        loss_proj = mask_losses_x + mask_losses_y  # cQ x Nins

        loss_pair = pairwise_loss_jit(src_masks, target_masks, image_lab_color, batch_i,
                                      self.boxvis_pairwise_color_thresh, num_masks, pair_stride)

        return {'loss_mask_proj': loss_proj, 'loss_mask_pair': loss_pair}

    def loss_masks_pseudo(self, outputs, targets, indices, num_masks, last_layer):
        src_idx = self._get_src_permutation_idx(indices)
        batch_i, src_i = src_idx
        src_masks = outputs["pred_masks"]

        src_masks = src_masks[src_idx]  # NxHpxWp
        tgt_masks_pseudo = torch.cat([t['masks_pseudo'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)  # NxHxW
        tgt_mask_scores_pseudo = torch.cat([t['mask_pseudo_scores'][i]
                                            for t, (_, i) in zip(targets, indices)]).to(src_masks)  # N

        # No need to align the size between predicted masks and ground-truth masks
        src_masks = src_masks[:, None]
        tgt_masks_pseudo = tgt_masks_pseudo[:, None]
        src_h, src_w = src_masks.shape[-2:]
        tgt_h, tgt_w = tgt_masks_pseudo.shape[-2:]

        is_high_conf = tgt_mask_scores_pseudo > self.mask_pseudo_score_thresh
        src_masks_high_conf = src_masks[is_high_conf]
        tgt_masks_pseudo_high_conf = tgt_masks_pseudo[is_high_conf]

        with torch.no_grad():
            # sample point_coords: N x 12544 x 2
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks_high_conf,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                tgt_masks_pseudo_high_conf,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks_high_conf,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        num_masks_high_conf = is_high_conf.sum().clamp(min=1)
        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks_high_conf),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks_high_conf),
        }

        use_boxvis_loss = True
        if use_boxvis_loss:
            # image_lab_color: 3xHxW, image_lab_color_masks: HxW
            tgt_masks_box = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)  # NxHxW
            tgt_masks_box = tgt_masks_box[:, None]
            image_lab_color = torch.cat([t["image_lab_color"][None] for t in targets]).to(src_masks)  # Bx3xHxW

            loss_type = 'downsample'
            # the predicted masks are inaccurate in the initial iterations
            if loss_type == 'upsample':
                pair_stride = 2
                src_masks = F.interpolate(src_masks, (tgt_h, tgt_w),
                                          mode='bilinear', align_corners=False)
            else:
                pair_stride = 1
                with torch.no_grad():
                    tgt_masks_box = F.interpolate(tgt_masks_box, (src_h, src_w),
                                                  mode='bilinear', align_corners=False).gt(0.5)
                    image_lab_color = F.interpolate(image_lab_color, (src_h, src_w),
                                                    mode='bilinear', align_corners=False)  # Bx3xHxW

            # ------------------ points out of bounding box / project term --------------------------
            # max operation may emphasize outlier segmentation in pseudo masks,
            # while average operation de-emphasize the outliers.
            mask_losses_y = dice_coefficient_loss_jit(
                src_masks.sigmoid().max(dim=-2, keepdim=True)[0],
                tgt_masks_box.max(dim=-2, keepdim=True)[0],
                num_masks
            )
            mask_losses_x = dice_coefficient_loss_jit(
                src_masks.sigmoid().max(dim=-1, keepdim=True)[0],
                tgt_masks_box.max(dim=-1, keepdim=True)[0],
                num_masks
            )
            loss_proj = mask_losses_x + mask_losses_y  # cQxNins

            loss_pair = pairwise_loss_jit(src_masks, tgt_masks_box, image_lab_color, batch_i,
                                          self.boxvis_pairwise_color_thresh, num_masks, pair_stride)
            losses.update({'loss_mask_proj': loss_proj, 'loss_mask_pair': loss_pair})

        del src_masks
        del src_masks_high_conf
        del tgt_masks_pseudo
        del tgt_masks_pseudo_high_conf

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, last_layer=False):
        if self.boxvis_enabled:
            if self.boxvis_ema_enabled:
                loss_map = {
                    'labels': self.loss_labels,
                    'masks': self.loss_masks_pseudo,
                }
            else:
                loss_map = {
                    'labels': self.loss_labels,
                    'masks': self.loss_masks_with_box_supervised,
                }
            if self.boxvis_boxdet_on:
                loss_map['boxes'] = self.loss_boxes
        else:
            loss_map = {
                'labels': self.loss_labels,
                'masks': self.loss_masks,
            }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, last_layer)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if self._iter <= self.boxvis_pairwise_warmup_iters:
            self._iter += 1

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, last_layer=True))

        fg_indices = []
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                fg_indices.append(aux_indices)  # intermediate layers

        fg_indices.append(indices)  # last layer
        return losses, fg_indices[-self.vita_last_layer_num:]

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
