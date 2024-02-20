# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample

from ..utils.box_ops import generalized_box_iou


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def batch_dice_coefficient_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = (inputs**2).sum(-1)[:, None] + (targets**2).sum(-1)[None, :]
    loss = 1 - numerator / denominator.clamp(min=1e-3)
    return loss


batch_dice_coefficient_loss_jit = torch.jit.script(
    batch_dice_coefficient_loss
)  # type: torch.jit.ScriptModule


def get_bounding_boxes(masks):
    """
    Returns:
        masks: NxHxW
        boxes: Nx4
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
    x_max = (x_max_val == 1).float() * (W - x_max)

    y_any = torch.any(masks, dim=2)
    y_any_cumsum = y_any.cumsum(dim=-1)
    y_any_cumsum[y_any_cumsum != 1] = H
    y_min = y_any_cumsum.argmin(dim=-1)

    y_any_cumsum = y_any[:, range(H)[::-1]].cumsum(dim=-1)
    y_any_cumsum[y_any_cumsum != 1] = H
    y_max_val, y_max = y_any_cumsum.min(dim=-1)
    y_max = (y_max_val == 1).float() * (H - y_max)

    boxes = torch.stack([x_min/float(W), y_min/float(H),
                         x_max/float(W), y_max/float(H)], dim=-1)

    return boxes


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, cost_box: float=1,
                 num_points: int = 0, boxvis_enabled: bool = False, boxvis_pairwise_size: int = 3,
                 boxvis_pairwise_dilation: int = 1, boxvis_pairwise_color_thresh: float = 0.3,
                 boxvis_boxdet_on: bool=False, boxvis_ema_enabled: bool=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points
        self.boxvis_enabled = boxvis_enabled
        self.boxvis_pairwise_size = boxvis_pairwise_size
        self.boxvis_pairwise_dilation = boxvis_pairwise_dilation
        self.boxvis_pairwise_color_thresh = boxvis_pairwise_color_thresh
        self.boxvis_boxdet_on = boxvis_boxdet_on

        self.boxvis_ema_enabled = boxvis_ema_enabled

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, tgt_mask_type='masks'):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]  # cQxK

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # cQxHpxWp
            # gt masks are already padded when preparing target
            tgt_mask = targets[b][tgt_mask_type].to(out_mask)  # cQxHxW

            out_mask = out_mask[:, None]  # cQx1xHpxWp
            tgt_mask = tgt_mask[:, None]  # cQx1xHxW
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss between masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def memory_efficient_box_supervised_forward(self, outputs, targets):
        bs, cQ, s_h, s_w = outputs["pred_masks"].shape

        indices = []
        for b in range(bs):
            b_out_prob = outputs["pred_logits"][b].softmax(-1)  # cQ x k
            tgt_ids = targets[b]["labels"]

            b_out_mask = outputs["pred_masks"][b][:, None]  # cQ x 1 x H_pred x W_pred
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(b_out_mask)[:, None]  # Nins x 1 x H_tgt x W_tgt

            if tgt_mask.shape[0] == 0:
                indices.append(([], []))
                continue

            # scores reflect the quality of predicted masks
            b_out_mask_soft = b_out_mask.sigmoid()
            b_out_mask_hard = b_out_mask_soft > 0.5
            numerator = (b_out_mask_soft.flatten(1) * b_out_mask_hard.flatten(1)).sum(1)
            denominator = b_out_mask_hard.flatten(1).sum(1)
            mask_scores = numerator / (denominator + 1e-6)  # cQ

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be omitted.
            cost_class = -b_out_prob[:, tgt_ids] * mask_scores.reshape(-1, 1)  # cQ x Nins

            # --------------------- project term --------------------------
            with autocast(enabled=False):
                tgt_mask_ds = F.interpolate(tgt_mask, (s_h, s_w),
                                            mode='bilinear', align_corners=False).gt(0.5)

                mask_losses_y = batch_dice_coefficient_loss_jit(
                    b_out_mask.max(dim=-2, keepdim=True)[0],
                    tgt_mask_ds.max(dim=-2, keepdim=True)[0].to(b_out_mask)
                )
                mask_losses_x = batch_dice_coefficient_loss_jit(
                    b_out_mask.max(dim=-1, keepdim=True)[0],
                    tgt_mask_ds.max(dim=-1, keepdim=True)[0].to(b_out_mask)
                )
                cost_proj = mask_losses_x + mask_losses_y  # cQxNins

                # box iou cost
                tgt_box = targets[b]["boxes"]
                if self.boxvis_boxdet_on:
                    b_out_box = outputs["pred_boxes"][b]
                    cost_box_sm = torch.cdist(b_out_box.flatten(1), tgt_box.flatten(1), p=1).to(cost_class)
                    cost_box_giou = 1 - generalized_box_iou(b_out_box, tgt_box)
                    cost_box = cost_box_sm + cost_box_giou
                else:
                    b_out_mask_box = get_bounding_boxes(b_out_mask.gt(0.).flatten(0, 1))  # cQx4
                    cost_mask_box_sm = torch.cdist(b_out_mask_box.flatten(1), tgt_box.flatten(1), p=1).to(cost_class)
                    cost_mask_biou = 1 - generalized_box_iou(b_out_mask_box, tgt_box)  # cQxNins
                    cost_box = cost_mask_box_sm + cost_mask_biou

            C = (
                    self.cost_dice * cost_proj
                    + self.cost_class * cost_class
                    + self.cost_box * cost_box * 0.5
            )
            C = C.reshape(cQ, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, cost_type='student'):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks
                 "cost_type": 'teacher' or 'student' when boxvis_ema_enabled

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if self.boxvis_enabled:
            if not self.boxvis_ema_enabled:
                return self.memory_efficient_box_supervised_forward(outputs, targets)
            else:
                if cost_type == 'teacher':
                    return self.memory_efficient_box_supervised_forward(outputs, targets)
                else:
                    return self.memory_efficient_forward(outputs, targets, tgt_mask_type='masks_pseudo')

        else:
            return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
