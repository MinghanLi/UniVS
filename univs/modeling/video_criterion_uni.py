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

from .video_criterion import VideoSetCriterion
from .video_criterion_prompt import VideoSetCriterionPrompt


class VideoSetCriterionUni(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, cfg, num_classes, matcher, weight_dict, eos_coef, losses):
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
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.disable_learnable_queries_sa1b = cfg.MODEL.UniVS.DISABLE_LEARNABLE_QUERIES_SA1B

        self.pixel_encoder_name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME # "MSDeformAttnPixelDecoderVL"
        self.prompt_as_queries = cfg.MODEL.UniVS.PROMPT_AS_QUERIES

        self.criterion = VideoSetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
            num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            use_ctt_loss=cfg.MODEL.UniVS.USE_CONTRASTIVE_LOSS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            max_num_masks=cfg.MODEL.UniVS.NUM_POS_QUERIES,
            # boxvis parameters
            boxvis_enabled=cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
        )

        if self.prompt_as_queries:
            self.criterion_prompt = VideoSetCriterionPrompt(
                num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=eos_coef,
                losses=losses,
                num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
                use_ctt_loss=cfg.MODEL.UniVS.USE_CONTRASTIVE_LOSS,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            )

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        if not self.prompt_as_queries:
            return self.criterion(outputs, targets)
        
        if self.disable_learnable_queries_sa1b and targets[0]["dataset_name"] == "sa_1b":
            if targets[0]["prompt_as_queries_enable"]:
                num_queries_l = 0
            else:  
                num_queries_l = self.num_queries
        else:
            num_queries_l = self.num_queries

        bs = len(targets)
        outputs_learnable = {}
        outputs_prompt = {}
        for k, v in outputs.items():
            if v is None:  # 'l2v_attn_weights' for parallel
                outputs_learnable[k] = v
                outputs_prompt[k] = v

            elif isinstance(v, list):
                if k == 'l2v_attn_weights':
                    outputs_prompt[k] = v  # a list or None
                    continue
                
                # aux output
                if k not in outputs_learnable:
                    outputs_learnable[k] = []
                    outputs_prompt[k] = []

                for v_aux_dict_per_layer in v:
                    outputs_learnable[k].append(
                        {k_: v_[:, :num_queries_l] for k_, v_ in v_aux_dict_per_layer.items() if k_ != "pred_reid_logits"}
                    )
                    outputs_prompt[k].append(
                        {k_: v_[:, num_queries_l:] for k_, v_ in v_aux_dict_per_layer.items() if k_ != "pred_reid_logits"}
                    )
                    # pred_reid_logits for uninext, which needs to be merged 
                    pred_reid_logits = v_aux_dict_per_layer["pred_reid_logits"]  # (BQT)(BQT) 
                    pred_reid_logits_l, pred_reid_logits_p, pred_reid_logits_l2p = \
                        self.process_reid_logits(pred_reid_logits, targets[0]['task'], bs, num_queries_l)
                    outputs_learnable[k][-1]["pred_reid_logits"] = pred_reid_logits_l
                    outputs_learnable[k][-1]["pred_reid_logits_l2p"] = pred_reid_logits_l2p
                    outputs_prompt[k][-1]["pred_reid_logits"] = pred_reid_logits_p

                    if "pred_embds" in outputs_prompt[k][-1]:
                        outputs_learnable[k][-1]["pred_embds_prompt"] = outputs_prompt[k][-1]["pred_embds"]

            else:
                if k == "pred_reid_logits":  # for uninext, which needs to be merged 
                    v_l, v_p, v_l2p = self.process_reid_logits(v, targets[0]['task'], bs, num_queries_l)
                    outputs_learnable[k] = v_l
                    outputs_learnable[k+"_l2p"] = v_l2p
                    outputs_prompt[k] = v_p
                else:
                    outputs_learnable[k] = v[:, :num_queries_l]
                    outputs_prompt[k] = v[:, num_queries_l:]

        if "pred_embds" in outputs_prompt:
            outputs_learnable["pred_embds_prompt"] = outputs_prompt["pred_embds"]
        
        losses = self.criterion(outputs_learnable, targets)
        losses_p = self.criterion_prompt(outputs_prompt, targets)
        for k, v in losses_p.items():
            if k in losses:
                losses[k] = 0.5 * (v + losses[k])
            else:
                losses[k] = v
        return losses
    
    def process_reid_logits(self, pred_reid_logits, task_type, bs, num_queries_l):
        pred_reid_logits = rearrange(pred_reid_logits, '(B Q T) (A P F) -> B Q T A P F', 
                                     B=bs, T=self.num_frames, A=bs, F=self.num_frames)
        # (BQ_lT) x (BQ_lT)
        pred_reid_logits_l = pred_reid_logits[:, :num_queries_l, :,:, :num_queries_l, :].flatten(0,2).flatten(1)
        # (BQ_pT) x (BQ_pT)
        pred_reid_logits_p = pred_reid_logits[:, num_queries_l:, :,:, num_queries_l:, :].flatten(0,2).flatten(1)
        # (BQ_lT) x (BQ_pT)
        pred_reid_logits_l2p = pred_reid_logits[:, :num_queries_l, :,:, num_queries_l:, :].flatten(0,2).flatten(1)
        return pred_reid_logits_l, pred_reid_logits_p, pred_reid_logits_l2p

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "num_frames: {}".format(self.num_frames),
            "eos_coef: {}".format(self.eos_coef),
            "prompt_as_queries: {}".format(self.prompt_as_queries)
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
