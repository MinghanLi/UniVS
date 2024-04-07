import os
import copy
import torch
import math

from torch import nn
from torch.nn import functional as F
from typing import Tuple
# support color space pytorch, https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html#rgb_to_lab
from kornia import color
from PIL import Image

import numpy as np
import pycocotools.mask as mask_util
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import batched_nms, box_area

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.utils.box_ops import box_xyxy_to_cxcywh
from univs import (
    VideoSetCriterionUni, 
    VideoHungarianMatcherUni,
    BoxVISTeacherSetPseudoMask,
    TextPromptEncoder,
    build_clip_language_encoder,
    Clips, 
    MDQE_OverTrackerEfficient,
    )
from univs.utils.comm import convert_mask_to_box, calculate_mask_quality_scores, box_iou, video_box_iou, batched_pair_mask_iou
from univs.prepare_targets import PrepareTargets

from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info

from .comm import match_from_learnable_embds, check_consistency_with_prev_frames
from .visualization import visualization_query_embds
from univs.utils.visualizer import VisualizerFrame

class InferenceVideoVOS(nn.Module):
    """
    Class for inference on video object segmentation task
    """

    @configurable
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        overlap_threshold_entity: float,
        stability_score_thresh: float,
        metadata,
        size_divisibility: int,
        LSJ_aug_image_size: int,
        LSJ_aug_enable_test: bool,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        num_frames: int,
        num_classes: int,
        data_name: str,
        # inference
        prompt_as_queries: bool,
        zero_shot_inference: bool,
        semantic_on: bool,
        instance_on: bool,
        panoptic_on: bool,
        test_topk_per_image: int,
        tracker_type: str,
        window_inference: bool,
        is_multi_cls: bool=True,
        apply_cls_thres: float=0.05,
        merge_on_cpu: bool=False,
        # tracking
        num_frames_window_test: int=5,
        clip_stride: int=1,
        output_dir: str='output/inf/vos/',
        temporal_consistency_threshold: float=0.25,
        video_unified_inference_queries: str='prompt',
        num_prev_frames_memory: int=5,
    ):
        """
        Args:
            num_queries: int, number of queries
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold
        self.overlap_threshold_entity = overlap_threshold_entity
        self.stability_score_thresh = stability_score_thresh
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.LSJ_aug_image_size = LSJ_aug_image_size
        self.LSJ_aug_enable_test = LSJ_aug_enable_test
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.num_classes = num_classes
        self.data_name = data_name
        self.is_coco = data_name.startswith("coco")
        # TODO: fixed number or dynamic number of masks in SA1B
        self.max_num_masks = 100  # there are more than 500 masks in sa1b, thereby balancing memory

        # additional args reference
        self.prompt_as_queries = prompt_as_queries
        self.zero_shot_inference = zero_shot_inference
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres
        self.window_inference = window_inference
        self.merge_on_cpu = merge_on_cpu
        
        # clip-by-clip tracking
        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"
        self.num_frames_window_test = max(num_frames_window_test, num_frames)
        self.clip_stride = clip_stride
        self.temporal_consistency_threshold = temporal_consistency_threshold
        self.video_unified_inference_queries = video_unified_inference_queries
        self.num_prev_frames_memory = max(num_prev_frames_memory, num_frames)

        self.output_dir = output_dir
        self.use_semseg_pvos = True

        self.visualize_results_only_enable = False
        self.visualize_query_emb_enable = False
        self.visualizer_query_emb = visualization_query_embds(
            reduced_type='pca',
            output_dir=output_dir,
        )
        
    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "overlap_threshold_entity": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD_ENTITY,
            "stability_score_thresh": cfg.MODEL.MASK_FORMER.TEST.STABILITY_SCORE_THRESH,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "LSJ_aug_image_size": cfg.INPUT.LSJ_AUG.IMAGE_SIZE,
            "LSJ_aug_enable_test": cfg.INPUT.LSJ_AUG.SQUARE_ENABLED,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "data_name": cfg.DATASETS.TEST[0],
            # inference
            "prompt_as_queries": cfg.MODEL.UniVS.PROMPT_AS_QUERIES,
            "zero_shot_inference": cfg.MODEL.BoxVIS.TEST.ZERO_SHOT_INFERENCE,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "tracker_type": cfg.MODEL.BoxVIS.TEST.TRACKER_TYPE,
            "window_inference": cfg.MODEL.BoxVIS.TEST.WINDOW_INFERENCE,
            "is_multi_cls": cfg.MODEL.BoxVIS.TEST.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.BoxVIS.TEST.APPLY_CLS_THRES,
            "merge_on_cpu": cfg.MODEL.BoxVIS.TEST.MERGE_ON_CPU,
            # tracking
            "num_frames_window_test": cfg.MODEL.BoxVIS.TEST.NUM_FRAMES_WINDOW,
            "clip_stride": cfg.MODEL.BoxVIS.TEST.CLIP_STRIDE,
            "output_dir": cfg.OUTPUT_DIR,
            "temporal_consistency_threshold": cfg.MODEL.UniVS.TEST.TEMPORAL_CONSISTENCY_THRESHOLD,
            "video_unified_inference_queries": cfg.MODEL.UniVS.TEST.VIDEO_UNIFIED_INFERENCE_QUERIES,
            "num_prev_frames_memory": cfg.MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def eval(self, model, batched_inputs):
        """
        Args:
            model: UniVS model
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]: each dict has the results for one image.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images_norm = [(x - self.pixel_mean) / self.pixel_std for x in images]

        if self.LSJ_aug_enable_test:
            padding_constraints = {"size_divisibility": self.size_divisibility, "square_size": self.LSJ_aug_image_size}
            images_norm = ImageList.from_tensors(images_norm, padding_constraints=padding_constraints)
        else:
            images_norm = ImageList.from_tensors(images_norm, self.size_divisibility)

        image_size = images_norm.image_sizes[0]
        interim_size = images_norm.tensor.shape[-2:]
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])
        out_size = (out_height, out_width)

        targets = model.prepare_targets.process_inference(
            batched_inputs, interim_size, self.device, model.text_prompt_encoder
        )
        targets[0]['video_len'] = len(images)

        self.inference_video_vos(model, batched_inputs, images_norm, targets, image_size, out_size)
    
    def inference_video_vos(self, model, batched_inputs, images, targets, image_size, out_size):
        images_tensor = images.tensor
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        video_len = len(images_tensor)
        
        is_last = False
        start_idx_window, end_idx_window = 0, 0
        stride = self.clip_stride
        for i in range(0, len(images_tensor), stride):
            if is_last and i + self.num_frames > video_len:
                break
            is_last = i + self.num_frames >= video_len
            targets[0]["frame_indices"] = torch.arange(i, min(i+self.num_frames, video_len))

            if i + self.num_frames > end_idx_window:
                start_idx_window, end_idx_window = i, min(i + self.num_frames_window_test, video_len)
                frame_idx_window = range(start_idx_window, end_idx_window)
                features_window = model.backbone(images_tensor[start_idx_window:end_idx_window])

            # step1: write the annotated masks for objects that firstly appear, and pad targets for all objects
            self.write_targets_into_annotations_per_clip(targets, i, stride)

            # step2: input images into model to obtain predictions
            features = {k: v[frame_idx_window.index(i):frame_idx_window.index(i)+self.num_frames]
                        for k, v in features_window.items()}
            out = model.sem_seg_head(features, targets=targets)
            del out['aux_outputs']

            # step3: write predictions into annotations, 
            # which can be used as the prompt of the following frames 
            self.write_predictions_into_annotations_per_clip(out, image_size, targets, i, stride)

            if self.visualize_results_only_enable:
                self.visualize_results(i, batched_inputs, targets, image_size, out_size, is_last, stride)
                continue

            if targets[0]['task'] == 'sot' or 'davis' in targets[0]['dataset_name']:
                self.save_vos_results(i, targets, image_size, out_size, is_last, stride)
            elif targets[0]['task'] == 'grounding':
                self.save_rvos_results(i, targets, image_size, out_size, is_last, stride)
            else:
                NotImplementedError
    
    def write_predictions_into_annotations_per_clip(self, out, image_size, targets, first_frame_idx, stride):
        """
        Write predictions per clip into annotations, which can be viewd as the pseudo ground-truth annotations
        and can be used to generate visual prompts of objects for the following frames
        Args:
            out: a Dict that stores all predicted masks, boxes
            image_size: image size without padding after data augmentation
            targets: A list with [Dict, Dict, ..], which stores the annotated masks of target-objects
            first_frame_idx: the indix of the first frame in the current processing clip
            stride: stride between two clips
        """
        # batch size is 1 here
        targets_per_video = targets[0]
        pred_logits = out['pred_logits'][0].float().sigmoid() # QxK
        pred_masks = out['pred_masks'][0].float()   # QxTxHxW
        pred_embds = out['pred_embds'][0].float()   # QxTxC
        h_pred, w_pred = pred_masks.shape[-2:]
        box_normalizer = torch.as_tensor([w_pred, h_pred, w_pred, h_pred], device=self.device)
        pred_boxes = convert_mask_to_box(pred_masks > 0) / box_normalizer.view(1,1,-1) 

        num_frames = pred_masks.shape[1]

        task = targets_per_video['task']
        if task == 'grounding':
            assert self.prompt_as_queries, 'only support prompts as queries for referring segmentation task'
        gt_masks = targets_per_video['masks']
        gt_mask_logits = targets_per_video['mask_logits']
        gt_boxes = targets_per_video['boxes']
        gt_embds = targets_per_video['embds']
        gt_labels = targets_per_video['labels']
        num_insts, _, h_gt, w_gt = gt_masks.shape
        
        pred_masks = F.interpolate(pred_masks, (h_gt, w_gt), mode='bilinear', align_corners=False)
        mask_quality_scores = calculate_mask_quality_scores(pred_masks[..., :image_size[0], :image_size[1]])
        if 'viposeg' in targets_per_video['dataset_name'] and self.use_semseg_pvos:
            num_classes, start_idx = combined_datasets_category_info['vipseg']
            pred_logits = pred_logits[..., start_idx:start_idx+num_classes]
            pred_logits_quality = pred_logits * mask_quality_scores.view(-1, 1)
            semseg = torch.einsum("qc,qthw->cthw", pred_logits_quality[:self.num_queries], pred_masks[:self.num_queries].sigmoid())
            sem_mask = semseg.argmax(0)

        # STEP1: firstly appeared objects
        first_appear_frame_idxs = targets_per_video["first_appear_frame_idxs"]
        is_first_appear = (first_appear_frame_idxs >= first_frame_idx) & (first_appear_frame_idxs < first_frame_idx+num_frames)
        if is_first_appear.any():
            faf_idx_ = first_appear_frame_idxs[is_first_appear] - (first_frame_idx + num_frames)
            obj_idx_  = torch.nonzero(is_first_appear).reshape(-1)

            use_prompt_only = True if task == "sot" else False # for first frame, we only use predicted masks by prompt queries
            if use_prompt_only or (self.prompt_as_queries and self.video_unified_inference_queries in {'prompt', 'prompt+learn', 'learn+prompt'}):
                # please enable LSJ_aug during inference (keep consistency postional embeddings with training)
                indices_p = obj_idx_ + self.num_queries

            gt_masks_first = gt_masks[obj_idx_, faf_idx_]
            gt_boxes_first = gt_boxes[obj_idx_, faf_idx_]
            if not use_prompt_only and self.video_unified_inference_queries in {'learn', 'prompt+learn', 'learn+prompt'}:
                # back-end re-identitfication between prompt and learnable queries, used in SEEM and UNINEXT
                # Box IoU to select topk candidates, Mask IoU to find the matched one
                biou_scores = video_box_iou(gt_boxes_first[:, None].repeat(1,num_frames,1), pred_boxes)[0]  # num_gt_first_occur x Q x T 
                biou_scores = biou_scores[range(is_first_appear.sum()), :, faf_idx_]  # num_gt_first_occur x Q
                num_topk = 5 
                topk_idxs = torch.topk(biou_scores, k=num_topk, dim=-1)[1] # num_gt_first_occur x num_topk
                pred_masks_topk = pred_masks[topk_idxs.flatten(), faf_idx_[:, None].repeat(1,num_topk).flatten()]
                pred_masks_topk = pred_masks_topk.reshape(-1, num_topk, h_gt, w_gt).gt(0.)
                miou_scores = batched_pair_mask_iou(gt_masks_first.unsqueeze(1).repeat(1,num_topk,1,1), pred_masks_topk)  
                indices_l = topk_idxs[range(is_first_appear.sum()), miou_scores.argmax(-1)]
            
            if use_prompt_only or (self.prompt_as_queries and self.video_unified_inference_queries == 'prompt'):
                # In first frame, only support prompts as queries for referring segmentation task
                matched_masks = pred_masks[indices_p]
                matched_mask_quality_scores = mask_quality_scores[indices_p]
                matched_pred_embds = pred_embds[indices_p]
                matched_pred_boxes = pred_boxes[indices_p]
            elif self.video_unified_inference_queries == 'learn':
                matched_masks = pred_masks[indices_l]
                matched_mask_quality_scores = mask_quality_scores[indices_l]
                matched_pred_embds = pred_embds[indices_l]
                matched_pred_boxes = pred_boxes[indices_l]
            else:
                w_p = mask_quality_scores[indices_p] / (mask_quality_scores[indices_p] + mask_quality_scores[indices_l]).clamp(min=1e-5)
                w_l = mask_quality_scores[indices_l] / (mask_quality_scores[indices_p] + mask_quality_scores[indices_l]).clamp(min=1e-5)
                matched_masks = w_p.view(-1,1,1,1) * pred_masks[indices_p] + w_l.view(-1,1,1,1) * pred_masks[indices_l]
                matched_mask_quality_scores = calculate_mask_quality_scores(matched_masks)
                matched_pred_embds = w_p.view(-1,1,1) * pred_embds[indices_p] + w_l.view(-1,1,1) * pred_embds[indices_l]
                matched_pred_boxes = w_p.view(-1,1,1) * pred_boxes[indices_p] + w_l.view(-1,1,1) * pred_boxes[indices_l]
            
            gt_embds[is_first_appear, -num_frames:] = matched_pred_embds
            if task == 'sot':
                is_bg = (matched_masks <= 0).all(0)
                matched_masks_weighted = matched_masks.sigmoid()
                miou_scores = batched_pair_mask_iou(
                    gt_masks_first.unsqueeze(1), 
                    matched_masks[range(is_first_appear.sum()), faf_idx_].gt(0.).unsqueeze(1)
                ).squeeze(1)
                matched_masks_weighted *= (miou_scores**2 * matched_mask_quality_scores).view(-1,1,1,1)
                matched_masks_ids = matched_masks_weighted.argmax(0)
                matched_masks_ids[is_bg] = -1
                matched_masks_binary = torch.stack([
                    matched_masks_ids == i for i in range(matched_masks.shape[0])
                ]).float()
                matched_masks = matched_masks * matched_masks_binary

                # remove low miou predictions 
                miou_scores = batched_pair_mask_iou(
                    gt_masks_first.unsqueeze(1), 
                    matched_masks_binary[range(is_first_appear.sum()), faf_idx_].unsqueeze(1)
                ).squeeze(1)
                mask_area = gt_masks_first.flatten(1).sum(1) / (96*96)
                is_above_miou = miou_scores > 0.15 * mask_area.clamp(max=1)
            else:
                is_above_miou = torch.ones(is_first_appear.sum())

            for i_, (is_above, obj_i_, faf_i_) in enumerate(zip(is_above_miou, obj_idx_, faf_idx_)):
                faf_i_ = faf_i_ + 1 if task == 'sot' else faf_i_
                cur_label = int(gt_labels[obj_i_])
                cur_mask = matched_masks[i_, faf_i_:]

                isstuff = False
                if 'viposeg' in targets_per_video['dataset_name']:
                    isstuff = cur_label + 1 in self.metadata.stuff_dataset_id_to_contiguous_id

                if (not is_above and not isstuff) or faf_i_ == 0:
                    continue
    
                # use semseg results to help stuff entities
                if 'viposeg' in targets_per_video['dataset_name'] and isstuff:
                    cur_mask[sem_mask[faf_i_:] == cur_label] = 10.
                gt_masks[obj_i_, faf_i_:] = cur_mask.gt(0.)
                gt_mask_logits[obj_i_, faf_i_:] = cur_mask
                gt_boxes[obj_i_, faf_i_:] = matched_pred_boxes[i_, faf_i_:]
        
        # STEP2: appeared objects
        has_appeared = (first_appear_frame_idxs < first_frame_idx) & (first_appear_frame_idxs != -1)
        if has_appeared.any():
            tgt_embds = gt_embds[has_appeared, -self.num_prev_frames_memory:]  # num_gt_appreaed x T_prev x C
            use_prompt = False
            if self.prompt_as_queries and self.video_unified_inference_queries in {'prompt', 'prompt+learn', 'learn+prompt'}:
                use_prompt = True
                sim_threshold = 0.5
                # please enable LSJ_aug during inference (keep consistency postional embeddings with training)
                indices_p = torch.nonzero(has_appeared).reshape(-1) + self.num_queries
                is_consistency, matched_sim_p = check_consistency_with_prev_frames(
                    tgt_embds, pred_embds[indices_p], sim_threshold=sim_threshold, return_similarity=True
                )
                
                matched_masks_p = pred_masks[indices_p]
                matched_mask_quality_scores_p = mask_quality_scores[indices_p]
                matched_pred_embds_p = pred_embds[indices_p]
                matched_pred_boxes_p = pred_boxes[indices_p]
                matched_masks_p[~is_consistency] = 0
                matched_mask_quality_scores_p[~is_consistency] = 0
                matched_pred_embds_p[~is_consistency] = 0
                matched_pred_boxes_p[~is_consistency] = 0
                matched_sim_p[~is_consistency] = 0

            use_learn = False
            if self.video_unified_inference_queries in {'learn', 'prompt+learn', 'learn+prompt'}:
                use_learn = True
                use_norm = 'viposeg' not in targets_per_video['dataset_name']
                sim_threshold = 0.65 if use_norm else 0.5
                # back-end re-identitfication between prompt and learnable queries, used in SEEM and UNINEXT
                indices_l, matched_sim_l = match_from_learnable_embds(
                    tgt_embds, pred_embds[:self.num_queries], 
                    return_similarity=True, return_src_indices=False, use_norm=use_norm
                )  # num_gt_appreaed
                matched_masks_l = pred_masks[indices_l]
                matched_mask_quality_scores_l = mask_quality_scores[indices_l]
                matched_pred_embds_l = pred_embds[indices_l]
                matched_pred_boxes_l = pred_boxes[indices_l]
                matched_logits_l = pred_logits[indices_l]

                is_consistency = matched_sim_l >= sim_threshold
                # if 'viposeg' in targets_per_video['dataset_name']:
                    # pred_labels = matched_logits_l.argmax(-1)
                    # for i_l, cur_label in enumerate(pred_labels):
                        # # a stuff object may include multiple regions (e.g. wall), 
                        # # the 1-to-1 assignment may miss some regions
                        # if cur_label + 1 in self.metadata.stuff_dataset_id_to_contiguous_id:
                        #     is_consistency[i_l] = 0
                matched_masks_l[~is_consistency] = 0
                matched_mask_quality_scores_l[~is_consistency] = 0
                matched_pred_embds_l[~is_consistency] = 0
                matched_pred_boxes_l[~is_consistency] = 0
                matched_sim_l[~is_consistency] = 0
            
            assert use_prompt or use_learn, 'Must use at least one of prompt or learn queries'
            if use_prompt and use_learn:
                matched_sim = (matched_sim_p + matched_sim_l) / (matched_sim_p.gt(0.).float() + matched_sim_l.gt(0.).float()).clamp(min=1)
                w_p = matched_sim_p / (matched_sim_p + matched_sim_l).clamp(min=1e-5)
                w_l = matched_sim_l / (matched_sim_p + matched_sim_l).clamp(min=1e-5)
                siou_up = (matched_masks_p.gt(0) & matched_masks_l.gt(0)).flatten(1).sum(1) 
                siou_dn = (matched_masks_p.gt(0) | matched_masks_l.gt(0)).flatten(1).sum(1)
                siou = siou_up / siou_dn.clamp(min=1)
                w_p[siou < 0.5] = 1
                w_l[siou < 0.5] = 0
                matched_masks = w_p.view(-1,1,1,1) * matched_masks_p + w_l.view(-1,1,1,1) * matched_masks_l
                matched_mask_quality_scores = calculate_mask_quality_scores(matched_masks)
                matched_pred_embds = w_p.view(-1,1,1) * matched_pred_embds_p + w_l.view(-1,1,1) * matched_pred_embds_l
                matched_pred_boxes = w_p.view(-1,1,1) * matched_pred_boxes_p + w_l.view(-1,1,1) * matched_pred_boxes_l
            elif use_prompt:
                matched_sim = matched_sim_p
                matched_masks = matched_masks_p
                matched_mask_quality_scores = matched_mask_quality_scores_p
                matched_pred_embds = matched_pred_embds_p
                matched_pred_boxes = matched_pred_boxes_p
            else:
                matched_sim = matched_sim_l
                matched_masks = matched_masks_l
                matched_mask_quality_scores = matched_mask_quality_scores_l
                matched_pred_embds = matched_pred_embds_l
                matched_pred_boxes = matched_pred_boxes_l
            
            if task == 'sot':
                original_area = (matched_masks > 0).flatten(1).sum(1).clamp(min=1)

                matched_masks_sigmoid = matched_masks.sigmoid()
                if 'viposeg' in targets_per_video['dataset_name'] and self.use_semseg_pvos:
                    for i, label in enumerate(gt_labels[has_appeared]):
                        if int(label) + 1 in self.metadata.stuff_dataset_id_to_contiguous_id:
                            matched_masks_sigmoid[i][sem_mask==label] = 1
                            matched_masks[i][sem_mask==label] = 10

                is_bg = (matched_masks <= 0).all(0)
                matched_masks_weighted = matched_masks_sigmoid * (matched_sim**2 * matched_mask_quality_scores).view(-1,1,1,1)
                matched_masks_ids = matched_masks_weighted.argmax(0)
                matched_masks_ids[is_bg] = -1
                matched_masks_binary = torch.stack([
                    matched_masks_ids == i for i in range(matched_masks.shape[0])
                ]).float()
            
                mask_area = matched_masks_binary.flatten(1).sum(1)
                above_ratio = (mask_area / original_area) > 0.25
                above_ratio = above_ratio & (original_area > 0) & (mask_area > 0)
                matched_masks_binary[~above_ratio] = 0.
                matched_masks = matched_masks * matched_masks_binary
            
            gt_mask_logits[has_appeared, -num_frames:] += matched_masks
            gt_boxes[has_appeared, -num_frames:] = matched_pred_boxes
            nonblank_embds = (gt_embds[has_appeared, -num_frames:] != 0).any(-1)
            gt_embds[has_appeared, -num_frames:] = \
                (gt_embds[has_appeared, -num_frames:] + matched_pred_embds) / (nonblank_embds[..., None] + 1.)

        targets_per_video['masks'] = gt_mask_logits.gt(0.).float()
        targets_per_video['mask_logits'] = gt_mask_logits
        targets_per_video['boxes'] = gt_boxes
        targets_per_video['embds'] = gt_embds
    
    def write_targets_into_annotations_per_clip(self, targets, first_frame_idx, stride):
        """
        Write annotated masks for these objects that appear in the first frame into the annotation Dict
        Args:
            targets: A list with [Dict, Dict, ..], which stores the annotated masks of target-objects
            first_frame_idx: the indix of the first frame in the current processing clip
        """
        for targets_per_video in targets:
            # Note: images without MEAN and STD normalization
            video_len = targets_per_video["video_len"]
            h_pad, w_pad = targets_per_video["inter_image_size"]
            box_normalizer = torch.as_tensor(
                [w_pad, h_pad, w_pad, h_pad], dtype=torch.float32, device=self.device
            ).reshape(1, -1)

            # init annotation for the first frame of the entire video
            if "ids" not in targets_per_video:   
                if targets_per_video['task'] == 'grounding':
                    _num_instance = len(targets_per_video["exp_obj_ids"])
                    targets_per_video["ids"] = [int(obj_id) for obj_id in targets_per_video["exp_obj_ids"]]
                    targets_per_video["first_appear_frame_idxs"] = torch.zeros(_num_instance, dtype=torch.long, device=self.device) 
                    targets_per_video["labels"] = torch.ones(_num_instance, dtype=torch.bool, device=self.device) * -1
                else:
                    gt_ids_per_video = list(set(sum([t.ori_ids for t in targets_per_video["instances"]], [])))
                    gt_ids_per_video = [gt_id for gt_id in gt_ids_per_video if gt_id != -1]
                    targets_per_video["ids"] = gt_ids_per_video
                    targets_per_video["first_appear_frame_idxs"] = torch.ones(len(gt_ids_per_video), dtype=torch.long, device=self.device) * -1
                    targets_per_video["labels"] = torch.ones(len(gt_ids_per_video), dtype=torch.bool, device=self.device) * -1
            targets_per_video["first_frame_idx"] = first_frame_idx

            # the last clip may have number frames that less than stride/num_frames
            num_frames = min(self.num_frames, video_len-first_frame_idx)

            _num_instance = len(targets_per_video["ids"])
            num_frames_newly = num_frames if first_frame_idx == 0 else min(stride, video_len-first_frame_idx)
              
            mask_shape = [_num_instance, num_frames_newly, h_pad, w_pad]
            gt_ids_per_video = targets_per_video["ids"]
            gt_classes_per_video = targets_per_video["labels"]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.float, device=self.device)
            gt_mask_logits_per_video = gt_masks_per_video.clone()
            gt_boxes_per_video = torch.zeros([_num_instance, num_frames_newly, 4], dtype=torch.float32, device=self.device)
            first_appear_frame_idxs = targets_per_video["first_appear_frame_idxs"]
            
            if first_frame_idx == 0:
                gt_embds_per_video = torch.zeros([_num_instance, num_frames_newly, self.hidden_dim], dtype=torch.float32, device=self.device)
            else:
                gt_embds_per_video = targets_per_video['embds'][:, -num_frames_newly:].mean(1).unsqueeze(1).repeat(1,num_frames_newly,1).clone()
                # remain masks of the last self.num_frames + 1 images for memory efficiency
                gt_masks_per_video = torch.cat([targets_per_video['masks'][:, -self.num_prev_frames_memory:], gt_masks_per_video], dim=1)  # N, num_frames, H, W
                gt_mask_logits_per_video = torch.cat([targets_per_video['mask_logits'][:, -self.num_prev_frames_memory:], gt_mask_logits_per_video], dim=1)  # N, num_frames, H, W
                gt_boxes_per_video = torch.cat([targets_per_video['boxes'], gt_boxes_per_video], dim=1)             # N, num_frames_prev, 4
                gt_embds_per_video = torch.cat([targets_per_video['embds'], gt_embds_per_video], dim=1)             # N, num_frames_prev, C

            if targets_per_video['task'] == 'sot':
                for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                    if f_i not in range(first_frame_idx, first_frame_idx + num_frames):
                        continue
                    
                    if len(targets_per_frame) == 0:
                        continue

                    targets_per_frame = targets_per_frame.to(self.device)
                    h, w = targets_per_frame.image_size

                    _update_id = [gt_ids_per_video.index(id) for id in targets_per_frame.ori_ids]
                    gt_boxes_per_video[_update_id, f_i] = targets_per_frame.gt_boxes.tensor / box_normalizer  # xyxy
                    _f_i = -(first_frame_idx + num_frames - f_i)
                    if isinstance(targets_per_frame.gt_masks, BitMasks):
                        gt_masks_per_video[_update_id, _f_i, :h, :w] = targets_per_frame.gt_masks.tensor.float()
                        gt_mask_logits_per_video[_update_id, _f_i, :h, :w] = targets_per_frame.gt_masks.tensor.float()
                    else:  # polygon
                        gt_masks_per_video[_update_id, _f_i, :h, :w] = targets_per_frame.gt_masks.float()
                        gt_mask_logits_per_video[_update_id, _f_i, :h, :w] = targets_per_frame.gt_masks.float()
                    # In vos dataset, there are some objects that appear in the intermediate frames
                    gt_classes_per_video[_update_id] = targets_per_frame.gt_classes
                    first_appear_frame_idxs[_update_id] = f_i
            
            targets_per_video.update(
                {
                    "labels": gt_classes_per_video, 
                    "masks": gt_masks_per_video,  # N, num_frames, H, W
                    "mask_logits": gt_mask_logits_per_video, # N, num_frames, H, W
                    "boxes": gt_boxes_per_video,  # N, num_frames_prev, 4
                    "embds": gt_embds_per_video,  # N, num_frames_prev, C
                    "first_appear_frame_idxs": first_appear_frame_idxs,
                }
            )
    
    def save_vos_results(self, first_frame_idx, targets, image_size, out_size, is_last, stride):
        # batch size is 1 here
        targets_per_video = targets[0]
        dataset_name = targets_per_video['dataset_name']
        video_len = len(targets_per_video['file_names'])
        video_id = targets_per_video['file_names'][0].split('/')[-2]
        file_names = targets_per_video['file_names'] 
        
        save_dir = os.path.join(self.output_dir, 'inference/Annotations', video_id)
        if first_frame_idx == 0:
            os.makedirs(save_dir, exist_ok=True)
        
        ids = torch.as_tensor(targets_per_video["ids"], device=self.device)
        if ids.min() == 0:
            ids += 1  # for grounding tasks
        pred_masks = targets_per_video['mask_logits']
        num_frames = min(self.num_frames, video_len-first_frame_idx)
        if not is_last:
            pred_masks = pred_masks[:, -num_frames:min(-num_frames+stride, -1)] # NTHW
        else:
            pred_masks = pred_masks[:, -num_frames:]

        pred_masks = pred_masks[:, :, :image_size[0], :image_size[1]]
        if image_size != out_size:
            pred_masks = F.interpolate(
                pred_masks.float(),
                out_size,
                mode='bilinear',
                align_corners=False
            )
        pred_masks = pred_masks.gt(0.).float()

        for t, m in enumerate(pred_masks.transpose(0, 1)):
            is_bg = (m <= 0).all(0)
            m = ids[m.argmax(0)]
            m[is_bg] = 0
        
            m = m.cpu().numpy().astype(np.uint8)
            m = Image.fromarray(m)
            m.putpalette(targets_per_video["mask_palette"])

            file_name = file_names[first_frame_idx+t].split('/')[-1]
            save_path = '/'.join([save_dir, file_name.replace('.jpg', '.png')])
            m.save(save_path)
            m.close()
        
        if is_last and self.visualize_query_emb_enable:
            self.visualizer_query_emb.visualization_query_embds(targets)
        # print('Saving predicted masks in', save_dir)
    
    def save_rvos_results(self, first_frame_idx, targets, image_size, out_size, is_last, stride):
        # batch size is 1 here
        targets_per_video = targets[0]
        video_len = len(targets_per_video['file_names'])
        video_name = targets_per_video['file_names'][0].split('/')[-2]
        file_names = targets_per_video['file_names'] 

        num_frames = min(self.num_frames, video_len-first_frame_idx)

        pred_masks = targets_per_video['mask_logits']
        if not is_last:
            pred_masks = pred_masks[:, -num_frames:min(-num_frames+stride, -1)] # NTHW
        else:
            pred_masks = pred_masks[:, -num_frames:]
        
        pred_masks = pred_masks[:, :, :image_size[0], :image_size[1]]
        ids = targets_per_video["ids"]
        if image_size != out_size:
            pred_masks = F.interpolate(
                pred_masks.float(),
                out_size,
                mode='bilinear',
                align_corners=False
            )
        pred_masks = pred_masks.gt(0.).float()

        for id_, mi in zip(ids, pred_masks):
            save_dir = os.path.join(self.output_dir, 'inference/Annotations', video_name, str(id_))
            os.makedirs(save_dir, exist_ok=True)

            for t, m in enumerate(mi):
                m = m * 255
                m = m.cpu().numpy().astype(np.uint8)
                m = Image.fromarray(m)

                file_name = file_names[first_frame_idx+t].split('/')[-1]
                save_path = '/'.join([save_dir, file_name.replace('.jpg', '.png')])
                m.save(save_path)
                m.close()
        
        if is_last and self.visualize_query_emb_enable:
            self.visualizer_query_emb.visualization_query_embds(targets)
        # print('Saving predicted masks in', save_dir)
    
    def visualize_results(self, first_frame_idx, batched_inputs, targets, image_size, out_size, is_last, stride):
        # batch size is 1 here
        targets_per_video = targets[0]
        video_len = len(targets_per_video['file_names'])
        video_name = targets_per_video['file_names'][0].split('/')[-2]
        file_names = targets_per_video['file_names'] 

        num_frames = min(self.num_frames, video_len-first_frame_idx)

        pred_masks = targets_per_video['masks']
        if not is_last:
            pred_masks = pred_masks[:, -num_frames:min(-num_frames+stride, -1)] # NTHW
        else:
            pred_masks = pred_masks[:, -num_frames:]
        
        pred_masks = pred_masks[:, :, :image_size[0], :image_size[1]]
        ids = targets_per_video["ids"]
        if image_size != out_size:
            pred_masks = F.interpolate(
                pred_masks.float(),
                out_size,
                mode='bilinear',
                align_corners=False
            ).gt(0.5).float()

        for t, m_t in enumerate(pred_masks.transpose(0, 1)):
            file_name = file_names[first_frame_idx+t].split('/')[-1]
            img = batched_inputs[0]["image"][first_frame_idx + t]
            img = F.interpolate(
                img[None].float(),
                out_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0).long()
            img = img.permute(1, 2, 0).cpu()

            for id_, m in enumerate(m_t):
                save_dir = os.path.join(self.output_dir, 'inference/Annotations', video_name, str(id_))
                os.makedirs(save_dir, exist_ok=True)
    
                visualizer = VisualizerFrame(
                    img, metadata=self.metadata, 
                )

                results = Instances(out_size)
                results.pred_masks = m.unsqueeze(0).cpu()
                results.scores = [1.]

                save_path = '/'.join([save_dir, file_name.replace('.png', '.jpg')])
                VisImage = visualizer.draw_instance_predictions(results)
                VisImage.save(save_path)

    
