# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample
)

from mask2former_video.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info


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
        num_masks: the average number of masks in the mini-batch

    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_wo_logits_loss(
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
        num_masks: the average number of masks in the mini-batch

    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_wo_logits_loss_jit = torch.jit.script(
    sigmoid_ce_wo_logits_loss
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
        num_masks: the average number of masks in the mini-batch

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
                The predictions for each example. Nx...
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: the average number of masks in the mini-batch
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = (inputs**2).sum(-1) + (targets**2).sum(-1)
    loss = 1 - numerator / denominator.clamp(min=1e-6)
    return loss.sum() / num_masks


dice_coefficient_loss_jit = torch.jit.script(
    dice_coefficient_loss
)  # type: torch.jit.ScriptModule


def focal_conf_sigmoid_loss(inputs, targets, alpha=0.5, gamma=2, is_cls=False):
    """
    Focal loss but using sigmoid like the original paper, where alpha balances the positive or negative samples,
    and gamma controls the easy or difficult samples.
    inputs: ... x N x K, masked these classes do not belong to this dataset
    targets: ... x N x k, one-hot embedding of targets
    Note: To make learnables mesh easier, the network predicts K+1 class confidences in this mode,
          but it has been masked with '-inf'.
    """
    # inputs = inputs.flatten(0, -2)    # N, num_classes
    # targets = targets.flatten(0, -2)  # N, num_classes

    targets_pm = targets * 2 - 1  # -1 if non-target classes, +1 if target class

    logpt = F.logsigmoid(inputs * targets_pm)  # note: 1 - sigmoid(x) = sigmoid(-x)
    pt = logpt.exp()

    # multi-class focal losses
    num_neg_classes = torch.logical_not(targets).sum(-1).clamp(min=1).unsqueeze(-1)
    at = alpha * targets + (1 - alpha) * (1 - targets)

    loss = -at * (1 - pt) ** gamma * logpt
    loss = loss.sum(-1)
    if not is_cls:
        loss = loss / targets.sum(-1).clamp(min=1)

    return loss

def contrastive_loss(inputs, targets, topk=50):
    if inputs.nelement() == 0:
        return inputs[:0].sum().detach()

    inputs = inputs.flatten(1)    # N, K
    targets = targets.flatten(1)  # N, K
    N = inputs.shape[0]

    pos_indices = targets.argmax(1)
    pos_inputs = inputs[torch.arange(N), pos_indices]

    pos_inputs_mean = (inputs * targets).sum(-1) / targets.sum(-1).clamp(min=1)
    pos_inputs = torch.stack([pos_inputs, pos_inputs_mean], dim=1)  # N K_pos
    
    neg_indices = torch.nonzero(targets.sum(0) > 0.).reshape(-1)
    bg_indices = torch.nonzero(targets.sum(0) == 0.).reshape(-1)
    neg_indices = neg_indices[torch.randperm(len(neg_indices))[:int(0.75*topk)]]
    bg_indices = bg_indices[torch.randperm(len(bg_indices))[:int(0.25*topk)]]
    neg_indices = torch.cat([neg_indices, bg_indices]).sort()[0]
    
    inputs = inputs[:, neg_indices]  # N K_neg
    targets = targets[:, neg_indices]  # N K_neg

    negpos_inputs = (inputs[:, :, None] - pos_inputs[:, None]) * torch.logical_not(targets)[:, :, None]  # N K_neg K_pos 
    negpos_inputs = negpos_inputs.clamp(max=10.).exp() * torch.logical_not(targets)[:, :, None]  # N K_neg K_pos
    
    # loss = torch.logsumexp(inputs_negpos, dim=-1)
    loss = (1 + torch.sum(negpos_inputs.flatten(1), dim=-1)).log()
    loss = loss.sum() / max(len(loss), 1)

    return loss

def contrastive_aux_loss(inputs, targets, topk=10):
    if inputs.nelement() == 0:
        return inputs[:0].sum().detach()
    
    # assert inputs.min() >= -1 and inputs.max() <= 1, f'invalid values: min {inputs.min()} and max {inputs.max()}'
    inputs = inputs.flatten(1)    # N, K
    targets = targets.flatten(1)  # N, K
    N = inputs.shape[0]

    pos_indices = torch.nonzero(targets.sum(0) > 0.).reshape(-1)
    bg_indices = torch.nonzero(targets.sum(0) == 0.).reshape(-1)
    bg_indices = bg_indices[torch.randperm(len(bg_indices))[:topk]]
    indices = torch.cat([pos_indices, bg_indices]).sort()[0]

    inputs = inputs[:, indices].clamp(min=0.)  # N K_neg
    targets = targets[:, indices]  # N K_neg

    return F.smooth_l1_loss(inputs, targets, reduction='sum') / max(len(inputs), 1)

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

class VideoSetCriterionPrompt(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_frames,
                 num_points, oversample_ratio, importance_sample_ratio, use_ctt_loss=True
                 ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_frames = num_frames
        self.use_ctt_loss = use_ctt_loss

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels_clip(self, outputs, targets, num_masks, l_layer):
        loss = []
        num_objs = []

        out_logits = outputs['pred_logits']
        for t, logits in zip(targets, out_logits):
            if logits.shape[0] == 0:
                continue 
            
            if t['task'] != 'grounding':
                if t['dataset_name'] not in combined_datasets_category_info:
                    continue
                num_classes, start_idx = combined_datasets_category_info[t['dataset_name']]
                logits = logits[:, start_idx:start_idx + num_classes]

                tgt_classes = torch.full(
                    logits.shape, 0, dtype=torch.int64, device=logits.device
                )  # NK

                is_appear = t["prompt_obj_ids"] >= 0
                if is_appear.sum() == 0:
                    continue

                tgt_idx = t["prompt_obj_ids"][is_appear]
                tgt_idx = tgt_idx.long()
                if t['prompt_type'] == 'text':
                    assert max(tgt_idx) < len(t["sem_labels"])
                    tgt_labels = t["sem_labels"][tgt_idx] - 1  # starts from 1
                else: # visual prompt
                    assert max(tgt_idx) < len(t["labels"])
                    tgt_labels = t["labels"][tgt_idx] - 1  # starts from 1
                src_idx = torch.nonzero(is_appear).reshape(-1)
                tgt_classes[src_idx, tgt_labels] = 1

                loss_focal = focal_conf_sigmoid_loss(logits, tgt_classes, is_cls=True)
                loss_focal = loss_focal.sum() / max(len(tgt_labels), 1)

                # cross-entroy loss
                if len(tgt_labels):
                    loss_ce = F.cross_entropy(logits[src_idx], tgt_labels)
                    loss.append(loss_focal + loss_ce)
                else:
                    loss.append(loss_focal)
                num_objs.append(len(tgt_labels))
            
            else:
                ids = t["prompt_obj_ids"]
                keep = ids >= 0
                ids[keep] = t["ids"].max(-1)[0][ids[keep]].long()

                ids_multihot = ((ids.unique().reshape(-1, 1) - ids.reshape(1, -1)) == 0).float()
                tgt_idx = [ids.unique().tolist().index(id_) for id_ in ids]
                tgt_classes = ids_multihot[tgt_idx]
                tgt_classes = tgt_classes[keep]
                logits = logits[keep]

                loss_focal = 0.2 * (
                    contrastive_loss(logits, tgt_classes) + contrastive_loss(logits.t(), tgt_classes.t())
                ) 
                loss.append(loss_focal)
                num_objs.append(keep.sum())
           
        if len(loss) == 0:
            return {"loss_ce": out_logits[:0].sum().detach()}
        else:
            weighted_loss = [num_obj / max(sum(num_objs), 1) * loss_ for num_obj, loss_ in zip(num_objs, loss)]
            return {"loss_ce": sum(weighted_loss)}
            # return {"loss_ce": sum(loss) / len(loss)}
    
    def loss_reid(self, outputs, targets, num_masks, l_layer):
        """ReID loss 
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes],
        we do not take the background class into account
        outputs["pred_embds"]: B x Q x T x C
        """
        assert "pred_embds" in outputs 
        if outputs["pred_embds"].nelement() == 0:
            return {
                "loss_reid": outputs["pred_embds"].sum().detach(),
                "loss_reid_aux": outputs["pred_embds"].sum().detach(),
            }
        
        device = outputs["pred_embds"].device
        bs = len(targets)
        
        pred_embds = outputs["pred_embds"].flatten(0, -2)                              # BQ_pT x c
        # tgt_ids = torch.stack([t["prompt_obj_ids"] for t in targets])
        # tgt_ids = tgt_ids[..., None].repeat(1,1,self.num_frames).to(device).flatten()  # BQ_pT
        tgt_ids = []
        for t in targets:
            valid = t["prompt_obj_ids"] >= 0
            tgt_ids_cur = t["prompt_obj_ids"][:, None].repeat(1, self.num_frames)
            tgt_ids_cur[valid] = t["ids"][t["prompt_obj_ids"][valid]].long()
            tgt_ids.append(tgt_ids_cur)
        tgt_ids = torch.stack(tgt_ids).to(device).flatten()  # BQ_pT
        
        vid_ids = torch.stack([
            torch.ones(len(t['prompt_obj_ids']), device=device)*i for i, t in enumerate(targets)
        ]).unsqueeze(-1)
        vid_ids = vid_ids.repeat(1,1,self.num_frames).flatten().to(device)  # BQ_pT

        keep = tgt_ids >= 0
        pred_embds = pred_embds[keep]
        tgt_ids = tgt_ids[keep]
        vid_ids = vid_ids[keep]

        tgt_classes = (tgt_ids[:, None] == tgt_ids[None]) & (vid_ids[:, None] == vid_ids[None])
        tgt_classes = tgt_classes.float()
        
        src_sim = torch.mm(pred_embds, pred_embds.t()) / math.sqrt(pred_embds.shape[-1])
        if not self.use_ctt_loss:
            loss_focal = focal_conf_sigmoid_loss(src_sim, tgt_classes, is_cls=False)
            loss_focal = loss_focal / max(self.num_frames * bs, 1)  # stable training
            loss_focal = loss_focal.sum() / max(loss_focal.nelement(), 1)
        else:
            loss_focal = contrastive_loss(src_sim, tgt_classes)
        
        # aux loss
        sim_aux = torch.einsum(
            'qc,kc->qk', F.normalize(pred_embds, p=2, dim=-1), 
            F.normalize(pred_embds, p=2, dim=-1)
        )

        sim_aux = sim_aux[tgt_classes.sum(-1) > 0]
        tgt_classes = tgt_classes[tgt_classes.sum(-1) > 0]
        loss_aux = contrastive_aux_loss(sim_aux, tgt_classes) 

        return  {"loss_reid": loss_focal, "loss_reid_aux": loss_aux}
        
    def loss_masks(self, outputs, targets, num_masks, l_layer):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, T, h, w]
        """
        assert "pred_masks" in outputs
        device = outputs["pred_masks"].device

        if outputs["pred_masks"].nelement() == 0: 
            src_masks = target_masks = outputs["pred_masks"][0]
        else:
            tgt_idx = torch.stack([t['prompt_obj_ids'] for t in targets]).to(device)  # BxQ_p
            keep = tgt_idx >= 0  
            tgt_idx = tgt_idx[keep]
            batch_idx, src_idx = torch.nonzero(keep).t()
            tgt_idx = [tgt_idx[batch_idx == i] for i in range(len(targets))]

            src_masks = outputs["pred_masks"][batch_idx, src_idx]
            if targets[0]["task"] == "detection" and targets[0]["prompt_type"] == "text":
                check_len = [t['sem_masks'].shape[0] > max(tgt_i) for t, tgt_i in zip(targets, tgt_idx) if len(tgt_i) > 0]
                assert False not in check_len
                target_masks = torch.cat([t['sem_masks'][tgt_i] for t, tgt_i in zip(targets, tgt_idx) if len(tgt_i)]).to(src_masks)  # NTHW
            else:
                target_masks = torch.cat([t['masks'][tgt_i] for t, tgt_i in zip(targets, tgt_idx)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords: NT x 12544 x 2
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

        # semantic mask loss 
        if targets[0]["task"] == 'detection' and targets[0]["prompt_type"] == "text":
            loss_ce_sem = self.loss_masks_sem(outputs, targets, num_masks, l_layer)
            losses["loss_mask"] = losses["loss_mask"] + loss_ce_sem
        if losses["loss_mask"].isnan().any():
            losses["loss_mask"] = losses["loss_mask"] * 0.  # odd error

        del src_masks
        del target_masks
        return losses
    
    def loss_masks_sem(self, outputs, targets, num_masks, l_layer):
        assert targets[0]["task"] == 'detection' and targets[0]["prompt_type"] == "text"
        device = outputs["pred_masks"].device

        src_masks = outputs["pred_masks"].transpose(1,2)  # BNTHW -> BTNHW
        if src_masks.nelement() == 0:
            return src_masks[:0].sum()

        # No need to upsample predictions as we are using normalized coordinates :)
        with torch.no_grad():
            tgt_masks = []
            for t in targets:
                is_appear = t["prompt_obj_ids"] >= 0
                if len(t['sem_masks']) == 0 or is_appear.sum() == 0:
                    sem_mask = torch.ones(t['sem_masks'].shape[1:], device=device) * -1
                else:
                    tgt_idx = t["prompt_obj_ids"].long()
                    is_bg = t['sem_masks'][tgt_idx].max(0)[0] == 0
                    sem_mask = t['sem_masks'][tgt_idx].max(0)[1].to(device)   # sem_ids: per pixel only belongs to per entity
                    sem_mask[is_bg] = -1
                tgt_masks.append(sem_mask)
            tgt_masks = torch.stack(tgt_masks).flatten(0,1).unsqueeze(1)  # (BT)1HW

            # sample point_coords: BT x 12544 x 2
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.flatten(0,1).max(1)[0].unsqueeze(1),
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                tgt_masks.float(),
                point_coords,
                mode='nearest',
                align_corners=False,
            ).flatten()  # BT * 12544
        
        src_masks = src_masks.flatten(0,1)
        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).transpose(1,2).flatten(0,1) # (BT)xNx12544 -> (BT*12544)xN

        keep = point_labels != -1  # remove padding area, whose lables is -1
        point_logits = point_logits[keep]
        point_labels = point_labels[keep]

        loss_ce = F.cross_entropy(point_logits, point_labels.long(), reduction='none')

        return loss_ce.sum() / max(1, point_labels.nelement())
    
    def loss_l2v_attn_weights(self, l2v_attn_weights, targets, num_masks):
        """Compute the losses related to the l2v_attn_weights, which is the same as mask loss.
        l2v_attn_weights: bs, nb_target_boxes, T, h, w
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, T, h, w]
        """
        assert l2v_attn_weights.dim() == 5
        device = l2v_attn_weights.device

        if (l2v_attn_weights.nelement() == 0) or (targets[0]["task"] == "detection" and targets[0]["prompt_type"] == "text"): 
            return {
                "loss_l2v_attn_weight": l2v_attn_weights[:0].sum().detach()
            }
        else:
            tgt_idx = torch.stack([t['prompt_obj_ids'] for t in targets]).to(device)  # BxQ_p
            keep = tgt_idx >= 0  
            tgt_idx = tgt_idx[keep]
            batch_idx, src_idx = torch.nonzero(keep).t()
            tgt_idx = [tgt_idx[batch_idx == i] for i in range(len(targets))]

            src_masks = l2v_attn_weights[batch_idx, src_idx]
            target_masks = torch.cat([t['masks'][tgt_i] for t, tgt_i in zip(targets, tgt_idx)]).to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # NT x 1 x H x W
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]

        with torch.no_grad():
            # sample point_coords: NT x 12544 x 2
            point_coords = get_uncertain_point_coords_with_randomness(
                0.9 - src_masks,
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

        point_probs = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        loss = F.smooth_l1_loss(point_probs, point_labels, reduction='none')
        loss = loss.sum() / max(loss.nelement(), 1)

        return {
            "loss_l2v_attn_weight": loss
        }

    def get_loss(self, loss, outputs, targets, num_masks, l_layer=9):
        loss_map = {
            'labels': self.loss_labels_clip,
            'masks': self.loss_masks,
            'reid': self.loss_reid,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_masks, l_layer)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item() * self.num_frames

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, num_masks, l_layer=9))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_masks, l_layer=i)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if 'l2v_attn_weights' in outputs:
            if outputs['l2v_attn_weights'] is None:
                loss_l2v = outputs_without_aux['pred_masks'][:0].sum().detach()
                losses.update({
                    "loss_l2v_attn_weight_0": loss_l2v, 
                    "loss_l2v_attn_weight_1": loss_l2v, 
                    "loss_l2v_attn_weight_2": loss_l2v
                }) 
            else:
                l2v_attn_weights_list = outputs['l2v_attn_weights']
                for i, l2v_attn_weights in enumerate(l2v_attn_weights_list):
                    l_dict = self.loss_l2v_attn_weights(l2v_attn_weights, targets, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
