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
from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info


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

    # -y*log(x)
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    # -(1-y)*log(1-x)
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


class VideoHungarianMatcherUni(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1,
                 cost_proj: float = 1, cost_pair: float = 1, num_points: int = 0,
                 boxvis_enabled: bool = False, boxvis_ema_enabled=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
            cost_proj: This is the relative weight of the projection loss of the binary mask in the matching cost
            cost_pair: This is the relative weight of the pairwise loss of the binary mask in the matching cost
            num_points: The number of sampling points to take part in the mask loss
            boxvis_enabled: It controls the annotation types: pixel-wise or box-level annotations for VIS task
            boxvis_ema_enabled: It controls whether to use Teacher Net to produce pseudo instance masks for VIS task
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_proj = cost_proj
        self.cost_pair = cost_pair

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

        # boxvis
        self.boxvis_enabled = boxvis_enabled
        self.boxvis_ema_enabled = boxvis_ema_enabled

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_masks"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_masks"][b]  # [num_queries, h, w]
            tgt_ids = targets[b]["labels"] - 1  # N

            if len(tgt_ids) == 0 or out_mask.nelement() == 0:
                indices.append(([], []))
                continue
            
            dataset_name = targets[b]["dataset_name"]
            if dataset_name in combined_datasets_category_info and "pred_logits" in outputs:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                num_classes, start_idx = combined_datasets_category_info[dataset_name]
                out_prob = outputs["pred_logits"][b][:, start_idx:start_idx+num_classes].sigmoid()
                out_prob = (out_prob * 5).softmax(-1)  # [num_queries, num_classes]
                cost_class = -out_prob[:, tgt_ids]
            else: 
                cost_class = 0

            out_mask = outputs["pred_masks"][b]  # cQxTxHpxWp
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]['masks'].to(out_mask)  # cQxTxHxW

            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

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
                + self.cost_dice * cost_dice
                + self.cost_class * cost_class
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def memory_efficient_box_supervised_forward(self, outputs, targets, dyn_pseudo_mask_score_thresh):
        bs, cQ, t, s_h, s_w = outputs["pred_masks"].shape

        indices = []
        for b in range(bs):
            b_out_prob = outputs["pred_logits"][b]  # cQ x k
            tgt_ids = targets[b]["labels"] - 1

            out_mask = outputs["pred_masks"][b]  # cQ x T x H_pred x W_pred
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # Nins x T x H_tgt x W_tgt

            if tgt_mask.shape[0] == 0:
                indices.append(([], []))
                continue

            # scores reflect the quality of predicted masks
            out_mask_soft = out_mask.sigmoid()
            out_mask_hard = out_mask_soft > 0.5
            numerator = (out_mask_soft.flatten(1) * out_mask_hard.flatten(1)).sum(1)
            denominator = out_mask_hard.flatten(1).sum(1)
            mask_scores = numerator / (denominator + 1e-6)  # cQ

            dataset_name = targets[b]["dataset_name"]
            if dataset_name in combined_datasets_category_info:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                num_classes, start_idx = combined_datasets_category_info[dataset_name]
                b_out_prob = (b_out_prob[:, start_idx:start_idx+num_classes].sigmoid() * 20).softmax(-1)  # [num_queries, num_classes]
                cost_class = -b_out_prob[:, tgt_ids] * mask_scores.reshape(-1, 1)  # cQ x Nins
            else: 
                cost_class = mask_scores.reshape(-1, 1)

            # --------------------- project term --------------------------
            with autocast(enabled=False):
                tgt_mask = F.interpolate(tgt_mask, (s_h, s_w),
                                         mode='bilinear', align_corners=False).gt(0.5)

                mask_losses_y = batch_dice_coefficient_loss_jit(
                    out_mask.max(dim=-2, keepdim=True)[0],
                    tgt_mask.max(dim=-2, keepdim=True)[0].to(out_mask)
                )
                mask_losses_x = batch_dice_coefficient_loss_jit(
                    out_mask.max(dim=-1, keepdim=True)[0],
                    tgt_mask.max(dim=-1, keepdim=True)[0].to(out_mask)
                )
                cost_proj = mask_losses_x + mask_losses_y  # cQxNins
            C = (
                    self.cost_proj * cost_proj
                    + self.cost_class * cost_class
            )

            if self.boxvis_ema_enabled and "masks_pseudo" in targets[b]:
                # pseudo mask supervision loss
                is_small_obj = (tgt_mask.flatten(1).sum(-1) / (t*s_h*s_w)) <= 0.05  # Nins
                # gt masks are already padded when preparing target
                tgt_mask_pseudo = targets[b]['masks_pseudo'].to(out_mask)  # cQxTxHxW
                tgt_mask_pseudo_score = targets[b]['mask_pseudo_scores'].to(out_mask)

                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # get gt labels
                tgt_mask_pseudo = point_sample(
                    tgt_mask_pseudo,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).flatten(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).flatten(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask_pseudo = tgt_mask_pseudo.float()
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask_pseudo)
                    low_conf = (tgt_mask_pseudo_score < min(dyn_pseudo_mask_score_thresh, 0.8)) | is_small_obj
                    cost_mask[:, low_conf] = 0.

                    # Compute the dice loss between masks
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask_pseudo)
                    cost_dice[:, low_conf] = 0

                C = C + self.cost_dice * cost_dice + self.cost_mask * cost_mask

            C = C.reshape(cQ, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, dyn_pseudo_mask_score_thresh=0.5):
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

            dyn_pseudo_mask_score_thresh: dynamic threshold to select high-quality pseudo masks on box-supervised VIS

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if self.boxvis_enabled:
            return self.memory_efficient_box_supervised_forward(outputs, targets, dyn_pseudo_mask_score_thresh)
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
