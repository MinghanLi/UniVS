import os
import cv2
import glob
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
# support color space pytorch, https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html#rgb_to_lab
from kornia import color

import numpy as np
import pycocotools.mask as mask_util
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

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
    FastOverTracker_DET,
    )
from univs.data.datasets import _get_vspw_vss_metadata, _get_vipseg_panoptic_metadata_val
from univs.utils.comm import convert_mask_to_box, calculate_mask_quality_scores, video_box_iou, batched_mask_iou
from univs.prepare_targets import PrepareTargets

from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info
from univs.utils.visualizer import VisualizerFrame
from .comm import (
    match_from_learnable_embds, 
    vis_clip_instances_to_coco_json_video, 
    check_consistency_with_prev_frames, 
    generate_temporal_weights
)

from .visualization import visualization_query_embds


class InferenceVideoEntity(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_queries: int,
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
        is_multi_cls: bool,
        apply_cls_thres: float,
        merge_on_cpu: bool,
        box_nms_thresh: float,
        # tracking
        num_max_inst_test: int,
        num_frames_window_test: int,
        clip_stride: int,
        output_dir: str,
        num_prev_frames_memory: int=5,
        video_unified_inference_entities: str='',
        temporal_consistency_threshold: float=0.5,
        detect_newly_object_threshold: float=0.05,
        detect_newly_interval_frames: int = 1,
        # custom videos
        custom_videos_enable: bool=False,
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
        self.box_nms_thresh = box_nms_thresh
        
        #  tracking
        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"
        self.num_max_inst_test = num_max_inst_test
        self.num_frames_window_test = max(num_frames_window_test, num_frames)
        self.num_frames_window_output = (math.ceil(self.num_frames_window_test / 5) + 1) * 5
        self.clip_stride = clip_stride

        self.use_quasi_track = True
        self.temporal_consistency_threshold = temporal_consistency_threshold
        self.detect_newly_object_threshold = detect_newly_object_threshold
        self.detect_newly_interval_frames = detect_newly_interval_frames
        self.num_prev_frames_memory = num_prev_frames_memory

        self.output_dir = output_dir
        self.custom_videos_enable = custom_videos_enable
        self.visualize_results_enable = True if custom_videos_enable else False
        self.visualizer = visualization_query_embds(
            reduced_type='pca',
            output_dir=output_dir,
        )

        if custom_videos_enable and len(video_unified_inference_entities) == 0:
            video_unified_inference_entities = 'entity_vps_vipseg'
        self.video_unified_inference_entities = video_unified_inference_entities
        if video_unified_inference_entities in {'entity_vss_entityseg', 'entity_vps_entityseg'}:
            self.metadata = MetadataCatalog.get('entityseg_panoptic_train')
        elif video_unified_inference_entities in {'entity_vss_vipseg', 'entity_vps_vipseg'}:
            self.metadata = MetadataCatalog.get("vipseg_panoptic_val")
        elif video_unified_inference_entities == 'entity_vis_entityseg':
            self.metadata = MetadataCatalog.get('entityseg_instance_train')
        elif video_unified_inference_entities == 'entity_vis_coco':
            self.metadata = MetadataCatalog.get('coco_2017_val')
        elif len(video_unified_inference_entities) and video_unified_inference_entities is not None:
            ValueError(f"Unsupported inference manner: {video_unified_inference_entities}")

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
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
            "box_nms_thresh": cfg.MODEL.UniVS.TEST.BOX_NMS_THRESH,
            # tracking
            "num_max_inst_test": cfg.MODEL.BoxVIS.TEST.NUM_MAX_INST,
            "num_frames_window_test": cfg.MODEL.BoxVIS.TEST.NUM_FRAMES_WINDOW,
            "clip_stride": cfg.MODEL.BoxVIS.TEST.CLIP_STRIDE,
            "output_dir": cfg.OUTPUT_DIR,
            # unified inference 
            "num_prev_frames_memory": cfg.MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY,
            "video_unified_inference_entities": cfg.MODEL.UniVS.TEST.VIDEO_UNIFIED_INFERENCE_ENTITIES,
            "temporal_consistency_threshold": cfg.MODEL.UniVS.TEST.TEMPORAL_CONSISTENCY_THRESHOLD,
            "detect_newly_object_threshold": cfg.MODEL.UniVS.TEST.DETECT_NEWLY_OBJECT_THRESHOLD,
            "detect_newly_interval_frames": cfg.MODEL.UniVS.TEST.DETECT_NEWLY_INTERVAL_FRAMES,
            # custom videos
            "custom_videos_enable": cfg.MODEL.UniVS.TEST.CUSTOM_VIDEOS_ENABLE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def eval(self, model, batched_inputs):
        """
        Args:
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

        interim_size = images_norm.tensor.shape[-2:]
        image_size = images_norm.image_sizes[0]
        targets = model.prepare_targets.process_inference(
            batched_inputs, interim_size, self.device, model.text_prompt_encoder, image_size
        )

        if len(self.video_unified_inference_entities):
            targets[0]["sub_task"] = self.video_unified_inference_entities
        else:
            dataset_name = targets[0]['dataset_name']
            if dataset_name.startswith("ytvis") or dataset_name.startswith("ovis"):
                targets[0]["sub_task"] = 'vis'
            elif dataset_name.startswith("vipseg"):
                targets[0]["sub_task"] = 'vps'
            elif dataset_name.startswith("vspw"):
                targets[0]["sub_task"] = 'vss'
            else:
                raise ValueError(f"Not support to eval the dataset {dataset_name} yet")

        return self.inference_video(model, batched_inputs, images_norm, targets)
        
    def inference_video(self, model, batched_inputs, images, targets):
        images_tensor = images.tensor
        sub_task = targets[0]["sub_task"]
        video_len = int(batched_inputs[0]["video_len"])
        
        # masks size
        interim_size = images_tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])
        out_size = (out_height, out_width)

        processed_results = []

        is_last = False
        start_idx_window, end_idx_window = 0, 0
        stride = self.num_frames if 'vss' in sub_task else self.clip_stride
        for i in range(0, len(images_tensor), stride):
            if is_last and (i + self.num_frames > len(images_tensor)):
                break

            is_last = (i + self.num_frames) >= len(images_tensor)
            targets[0]["first_frame_idx"] = i
            targets[0]["frame_indices"] = torch.arange(i, min(i+self.num_frames, len(images_tensor)))

            if i + self.num_frames > end_idx_window:
                start_idx_window, end_idx_window = i, i + self.num_frames_window_test
                frame_idx_window = range(start_idx_window, end_idx_window)
                features_window = model.backbone(images_tensor[start_idx_window:end_idx_window])

            features = {k: v[frame_idx_window.index(i):frame_idx_window.index(i)+self.num_frames]
                        for k, v in features_window.items()}
            out = model.sem_seg_head(features, targets=targets)
            del out['aux_outputs']

            # map logits into [0, 1]
            out['pred_logits'] = out['pred_logits'].sigmoid()
            if sub_task.startswith('entity'):
                if sub_task in {'entity_vss_entityseg', 'entity_vps_entityseg'}:
                    dataset_name = 'entityseg_panoptic'  # 644 classes
                elif sub_task in {'entity_vss_vipseg', 'entity_vps_vipseg'}:
                    dataset_name = 'vipseg'
                elif sub_task == 'entity_vis_entityseg': 
                    dataset_name = "entityseg_instance"  # 206 classes 
                elif sub_task == 'entity_vis_coco':
                    sdataset_name = 'coco'
                else:
                    raise ValueError
                num_classes, start_idx = combined_datasets_category_info[dataset_name]
                assert start_idx + num_classes <= out['pred_logits'].shape[-1]
                out['pred_logits'] = out['pred_logits'][..., start_idx:start_idx + num_classes]
            else:
                dataset_name = targets[0]['dataset_name']
                if dataset_name in combined_datasets_category_info:
                    num_classes, start_idx = combined_datasets_category_info[dataset_name]
                    assert start_idx + num_classes <= out['pred_logits'].shape[-1]
                    out['pred_logits'] = out['pred_logits'][..., start_idx:start_idx + num_classes]
            
            # remove bacth dim here
            for k, v in out.items():
                if v is None:
                    continue
                out[k] = v[0]
                
            out_learn, out_prompt = {}, {}
            for k, v in out.items():
                if v is None:
                    continue
                out_learn[k] = v[:self.num_queries]
                out_prompt[k] = v[self.num_queries:]
            del out

            if 'vss' in sub_task:
                processed_results.append(
                    self.save_results_vss(i, out_learn, interim_size, image_size, out_size, is_last, stride)
                )

            elif 'vis' in sub_task or 'vps' in sub_task:
                # step1: update annotations for prompt-specified entities
                self.write_prompt_predictions_into_annotations_per_clip(i, out_prompt, targets, interim_size, image_size, stride)

                # step2: find newly entities 
                if i % self.detect_newly_interval_frames == 0 or targets[0]["masks"].nelement() == 0:
                    if 'vis' in sub_task:
                        self.detect_newly_entities_per_clip_instance(out_learn, targets, interim_size)
                    elif 'vps' in sub_task:
                        self.detect_newly_entities_per_clip_pixel(out_learn, targets, interim_size)
                    else:
                        raise ValueError

                    # convert newly entities into prompt-specified annotations
                    self.write_newly_entities_into_annotations_per_clip(i, out_learn, targets, interim_size)

                # step3: save results
                is_out = i > self.num_prev_frames_memory and i % self.num_frames_window_output == self.num_prev_frames_memory
                if is_out or is_last:
                    if 'vis' in sub_task:
                        processed_results.append(
                            self.save_results_vis(i, targets, interim_size, image_size, out_size, is_last)
                        )
                    elif 'vps' in sub_task:
                        processed_results.append(
                            self.save_results_vps(i, targets, interim_size, image_size, out_size, is_last)
                        )
                    else:
                        raise ValueError
                    
                    if self.visualize_results_enable and 'vis' in sub_task:
                        self.visualize_results_vis(i, batched_inputs, targets, image_size, out_size, is_last)
                    
                    # remove previous masks in memory pool for memoty efficiently
                    targets[0]["mask_logits"] = targets[0]["mask_logits"][:, self.num_frames_window_output:]
                    targets[0]["masks"] = targets[0]["masks"][:, self.num_frames_window_output:]
                    targets[0]["occurrence"] = targets[0]["occurrence"][:, self.num_frames_window_output:]

            else:
                raise ValueError(f"Not support to eval the dataset {dataset_name} yet")

            # pad zero values for prompt-specified annotations
            if not is_last and "masks" in targets[0]:
                self.pad_zero_annotations_for_next_clip(targets, min(stride, video_len-i-self.num_frames))

        if 'vis' in sub_task:
            if self.visualize_results_enable:
                self.visualizer.visualization_query_embds(targets)

            return vis_clip_instances_to_coco_json_video(
                batched_inputs, processed_results, test_topk_per_video=self.test_topk_per_image
            )
        elif 'vps' in sub_task:
            if self.visualize_results_enable:
                # visualize any video in vps format
                processed_results = self.vps_output_results(targets, processed_results, out_size)
                self.visualize_results_vps(batched_inputs, processed_results, out_size, sub_task)
                return []
            else:
                 # evaluation metrics
                return self.vps_output_results(targets, processed_results, out_size)
        elif 'vss' in sub_task:
            if self.visualize_results_enable:
                # visualize any video in vss format
                processed_results = self.vss_output_results(targets, processed_results, out_size)
                self.visualize_results_vss(batched_inputs, processed_results, out_size, sub_task)
                return []
            else:
                # evaluation metrics for vspw 
                return self.vss_output_results(targets, processed_results, out_size)
        else:
            raise ValueError(f"Not support to eval the dataset {dataset_name} yet")
    
    def write_prompt_predictions_into_annotations_per_clip(self, first_frame_idx, out, targets, interim_size, image_size, stride):
        """
        Write predictions per clip into annotations, which can be viewd as the pseudo ground-truth annotations
        and can be used to generate visual prompts of objects for the following frames
        Args:
            out: a Dict that stores all predicted masks, boxes
            targets: A list with [Dict, Dict, ..], which stores the annotated masks of target-objects
            first_frame_idx: the indix of the first frame in the current processing clip
        """
        if out['pred_masks'].nelement() == 0:
            # there is no prompt queries
            return

        # batch size is 1 here
        pred_logits = out['pred_logits'] # Q_pxK
        pred_masks = out['pred_masks']   # Q_pxTxHxW
        pred_embds = out['pred_embds']   # Q_pxTxC
        pred_masks = F.interpolate(
            pred_masks, interim_size, mode='bilinear', align_corners=False
        )

        num_frames = pred_masks.shape[1]

        targets_per_video = targets[0]
        gt_logits = targets_per_video['logits']
        gt_masks = targets_per_video['masks']
        gt_mask_logits = targets_per_video['mask_logits']
        gt_boxes = targets_per_video['boxes']
        gt_embds = targets_per_video['embds']
        gt_occurrence = targets_per_video['occurrence']
        gt_mask_quality_scores = targets_per_video["mask_quality_scores"]
        dataset_name = targets_per_video['dataset_name']
        
        # check the consistency of entity masks with thh previous frames
        temporal_consistency_threshold = self.temporal_consistency_threshold 
        if first_frame_idx < self.num_frames:
            temporal_consistency_threshold *= 0.5
        is_consistency, sim_consistency = check_consistency_with_prev_frames(
            gt_embds[:, -max(int(self.num_prev_frames_memory/stride), 3):], pred_embds, 
            sim_threshold=temporal_consistency_threshold, return_similarity=True
        )

        cur_masks = pred_masks[:, :, :image_size[0], :image_size[1]]
        mask_quality_scores = calculate_mask_quality_scores(cur_masks) 
        if 'vis' in targets[0]["sub_task"]:
            # process overlapped area by multiple masks
            cur_scores = gt_logits.mean(1).max(-1)[0] * sim_consistency * mask_quality_scores
            cur_masks = cur_masks.sigmoid().flatten(1)
            is_bg = (cur_masks < 0.5).sum(0) == len(cur_masks)
            cur_prob_masks = cur_scores.view(-1, 1) * cur_masks
            cur_mask_ids = cur_prob_masks.argmax(0)  # take argmax (t, h, w)
            cur_mask_ids[is_bg] = -1
            cur_mask_ids = (cur_mask_ids[None] - torch.arange(cur_masks.shape[0], device=cur_masks.device).view(-1,1)) == 0

            original_area = (cur_masks > 0.5).sum(1).clamp(min=1)
            mask_area = cur_mask_ids.sum(1)
            above_ratio = (mask_area / original_area) > self.overlap_threshold_entity
            mask_over = (cur_mask_ids & (cur_masks > 0.5)).sum(1) > 0
            is_consistency = is_consistency & above_ratio & mask_over 

        if is_consistency.sum():
            matched_masks = pred_masks[is_consistency]
            box_normalizer = torch.as_tensor([interim_size[1], interim_size[0], interim_size[1], interim_size[0]], device=self.device)
            
            # gt_logits[is_consistency, -1] = pred_logits[is_consistency]
            nonblank_masks = matched_masks.flatten(-2).gt(0.).any(-1)
            gt_occurrence[is_consistency, -num_frames:] += nonblank_masks.float() 
            gt_mask_logits[is_consistency, -num_frames:] += matched_masks.clone() 
            gt_boxes[is_consistency, -num_frames:] = \
                convert_mask_to_box(gt_mask_logits[is_consistency, -num_frames:] > 0) / box_normalizer.view(1,1,-1)
            nonblank_embds = (gt_embds[is_consistency, -1] != 0).any(-1)
            gt_embds[is_consistency, -1] = \
                (gt_embds[is_consistency, -1] + pred_embds[is_consistency].mean(1)) / (nonblank_embds[..., None] + 1.)

            gt_mask_quality_scores[is_consistency] += mask_quality_scores[is_consistency] 
            
        targets_per_video['logits'] = gt_logits
        targets_per_video['masks'] = gt_mask_logits.gt(0.).float()
        targets_per_video['mask_logits'] = gt_mask_logits
        targets_per_video['boxes'] = gt_boxes
        targets_per_video['embds'] = gt_embds
        targets_per_video['occurrence'] = gt_occurrence
        targets_per_video["mask_quality_scores"] = gt_mask_quality_scores
    
    def detect_newly_entities_per_clip_instance(self, out_learn, targets, interim_size):
        is_first_frame = "masks" not in targets[0]

        # remove duplicated entitis in per clip
        pred_logits = out_learn['pred_logits'].float() # Q_lxK
        pred_masks = out_learn['pred_masks'].float()   # Q_lxTxHxW
        pred_embds = out_learn['pred_embds'].float()   # Q_lxTxC 

        num_frames = pred_masks.shape[1]

        mask_quality_scores = calculate_mask_quality_scores(pred_masks)
        pred_logits = pred_logits * mask_quality_scores.view(-1, 1)
        if self.stability_score_thresh > 0.:
            keep = mask_quality_scores > self.stability_score_thresh
            pred_logits = pred_logits[keep]
            pred_masks = pred_masks[keep]
            pred_embds = pred_embds[keep]
            mask_quality_scores = mask_quality_scores[keep]
        
        nms_scores, nms_labels = pred_logits.max(-1)
        sorted_scores, sorted_indices = nms_scores.sort(descending=True)
        keep = sorted_indices[:self.test_topk_per_image]
        pred_logits = pred_logits[keep]
        pred_masks = pred_masks[keep]
        pred_embds = pred_embds[keep]
        nms_scores = nms_scores[keep]
        nms_labels = nms_labels[keep]
        mask_quality_scores = mask_quality_scores[keep]

        h_pred, w_pred = pred_masks.shape[-2:]
        box_normalizer = torch.as_tensor([w_pred, h_pred, w_pred, h_pred], device=self.device)
        pred_boxes = convert_mask_to_box(pred_masks > 0) / box_normalizer
        if pred_masks.shape[0] > 1:
            # NMS to remove reduplicated masks in the first frame
            sorted_indices = nms_scores.sort(descending=True)[1]
            use_biou = True
            if use_biou:
                biou = video_box_iou(pred_boxes[sorted_indices], pred_boxes[sorted_indices])[0]
                max_biou = biou.max(-1)[0]  # N_gt, Q_l, T -> N_gt, Q_l
                max_biou = torch.triu(max_biou, diagonal=1).max(0)[0]
                keep_by_nms = sorted_indices[max_biou < self.box_nms_thresh]
                
            else:
                 # Calculate rank of query embeds
                pred_embds_one_hot = (pred_embds.mean(1)*1000).softmax(-1)
                emb_one_hot_rank = np.linalg.matrix_rank(pred_embds_one_hot.cpu().numpy())
                # print(f"The rank of the matrix with shape {pred_embds.shape} is: {emb_one_hot_rank}")
                emb_one_hot_sim = torch.mm(pred_embds_one_hot[sorted_indices], pred_embds_one_hot[sorted_indices].t())
                max_sim = torch.triu(emb_one_hot_sim, diagonal=1).max(0)[0]
                topk_sim, topk_indices = torch.topk(1-max_sim, k=emb_one_hot_rank)
                # print(1-topk_sim)
                keep_by_nms = sorted_indices[topk_indices]
            
            pred_logits = pred_logits[keep_by_nms]
            pred_masks = pred_masks[keep_by_nms]
            pred_embds = pred_embds[keep_by_nms]  # Q_l, T, C
            pred_boxes = pred_boxes[keep_by_nms]
            mask_quality_scores = mask_quality_scores[keep_by_nms]
        
        if is_first_frame:
            # sotre entities with high confidence scores in the first frame
            newly_indices = pred_logits.max(-1)[0] > max(self.apply_cls_thres, 0.1)
        else:
            targets_per_video = targets[0]
            # find newly entities in the subsequent frames
            gt_logits = targets_per_video['logits']
            gt_masks = targets_per_video['masks']
            gt_mask_logits = targets_per_video['mask_logits']
            gt_boxes = targets_per_video['boxes']  # N_gt, T_prev, 4
            gt_embds = targets_per_video['embds']  # N_gt, T_prev, C
            gt_occurrence = targets_per_video['occurrence']
            gt_mask_quality_scores = targets_per_video["mask_quality_scores"]

            tgt_embds = gt_embds[:, -3:]
            if self.use_quasi_track:
                sim = torch.einsum('ntc,mfc->nmtf', tgt_embds, pred_embds).flatten(2)  # N_gt, N_pred, K
                sim_bi = (sim.softmax(1) + sim.softmax(0)).mean(-1) / 2. 
                sim_bi[sim_bi < self.detect_newly_object_threshold] = 0
                indices = linear_sum_assignment((1 - sim_bi).cpu())
                matched_sim = sim_bi[indices]
            else:
                indices, matched_sim = match_from_learnable_embds(
                    tgt_embds, pred_embds, return_similarity=True, return_src_indices=True, 
                    use_norm=True, thresh=self.detect_newly_object_threshold
                )
                
            above_sim = matched_sim > self.detect_newly_object_threshold
            matched_tgt_indices = torch.as_tensor(indices[0], device=matched_sim.device)[above_sim]
            matched_pred_indices = torch.as_tensor(indices[1], device=matched_sim.device)[above_sim]
            # !!Important: Must update GT embds from learnable queries, mismatch feature spaces yet
            gt_logits[matched_tgt_indices, -1] = 0.5 * (gt_logits[matched_tgt_indices, -1] + pred_logits[matched_pred_indices])
            nonblank_embds = (gt_embds[matched_tgt_indices, -1] != 0).any(-1)
            gt_embds[matched_tgt_indices, -1] = \
                (gt_embds[matched_tgt_indices, -1] + pred_embds[matched_pred_indices].mean(1)) / (nonblank_embds[..., None] + 1.)
            
            targets_per_video['logits'] = gt_logits
            targets_per_video['embds'] = gt_embds

            # update the detected entities stored in the memory pool
            update_masks_from_learnable = True
            if update_masks_from_learnable:
                above_sim = matched_sim > 2*self.detect_newly_object_threshold
                matched_tgt_indices = torch.as_tensor(indices[0], device=matched_sim.device)[above_sim]
                matched_pred_indices = torch.as_tensor(indices[1], device=matched_sim.device)[above_sim]
                matched_masks = F.interpolate(
                    pred_masks[matched_pred_indices], interim_size, mode='bilinear', align_corners=False
                )
                nonblank_masks = matched_masks.flatten(-2).gt(0.).any(-1)
                gt_occurrence[matched_tgt_indices, -num_frames:] += nonblank_masks.float()
                gt_mask_logits[matched_tgt_indices, -num_frames:] += matched_masks.clone() 
                gt_mask_quality_scores[matched_tgt_indices] += mask_quality_scores[matched_pred_indices]

                targets_per_video['mask_logits'] = gt_mask_logits
                targets_per_video['masks'] = gt_mask_logits.gt(0.).float()
                targets_per_video['occurence'] = gt_occurrence
                targets_per_video["mask_quality_scores"] = gt_mask_quality_scores
            
            # detect newly entities
            newly_indices = []
            gt_mask_logits_ds = F.interpolate(
                gt_mask_logits[:, -num_frames:], pred_masks.shape[-2:], mode='bilinear', align_corners=False
            )
            for idx in range(pred_embds.shape[0]):
                s_, l_ = pred_logits[idx].max(-1)
                if idx not in matched_pred_indices and s_ > self.apply_cls_thres:
                    miou = batched_mask_iou(pred_masks[idx][:, None].gt(0.), gt_mask_logits_ds.transpose(0, 1).gt(0.))
                    if miou.nelement() and miou.max() < 0.5:
                        newly_indices.append(idx)
            
        out_learn['pred_logits'] = pred_logits[newly_indices]
        out_learn['pred_masks'] = pred_masks[newly_indices]
        out_learn['pred_embds'] = pred_embds[newly_indices]
        out_learn['pred_boxes'] = pred_boxes[newly_indices]
        out_learn['mask_quality_scores'] = mask_quality_scores[newly_indices]
    
    def detect_newly_entities_per_clip_pixel(self, out_learn, targets, interim_size):
        is_first_frame = "masks" not in targets[0]

        # remove duplicated entitis in per clip
        pred_logits = out_learn['pred_logits'].float() # Q_lxK
        pred_masks = out_learn['pred_masks'].float()   # Q_lxTxHxW
        pred_embds = out_learn['pred_embds'].float()   # Q_lxTxC 
        h_pred, w_pred = pred_masks.shape[-2:]
        box_normalizer = torch.as_tensor([w_pred, h_pred, w_pred, h_pred], device=self.device)
        pred_boxes = convert_mask_to_box(pred_masks > 0) / box_normalizer

        num_frames = pred_masks.shape[1]

        mask_quality_scores = calculate_mask_quality_scores(pred_masks)
        pred_logits = pred_logits * mask_quality_scores.view(-1, 1)
        nms_scores, nms_labels = pred_logits.max(-1)

        if is_first_frame:
            # NMS to remove reduplicated masks in the first frame
            sorted_indices = nms_scores.sort(descending=True)[1][:100]
            sorted_labels = nms_labels[sorted_indices] + 1  # category labels start from 1
            assert self.metadata.thing_dataset_id_to_contiguous_id is not None
            isthing = torch.as_tensor([int(l) in self.metadata.thing_dataset_id_to_contiguous_id for l in sorted_labels])
            sorted_indices_thing = sorted_indices[isthing]
            sorted_indices_stuff = sorted_indices[~isthing]
    
            if len(sorted_indices_thing):
                sorted_indices_thing = sorted_indices_thing[:70]
                biou = video_box_iou(pred_boxes[sorted_indices_thing], pred_boxes[sorted_indices_thing])[0]
                max_biou = biou.max(-1)[0]  # N_gt, Q_l, T -> N_gt, Q_l
                max_biou = torch.triu(max_biou, diagonal=1).max(0)[0]
                sorted_indices_thing = sorted_indices_thing[max_biou < self.box_nms_thresh]
            
            if len(sorted_indices_stuff):
                sorted_indices_stuff = sorted_indices_stuff[:30]
                pred_masks_stuff = pred_masks[sorted_indices_stuff][:, 0].gt(0.).float().unsqueeze(0)
                max_miou = batched_mask_iou(pred_masks_stuff, pred_masks_stuff).max(0)[0]  # T, N_gt, Q -> N_gt, Q
                max_miou = torch.triu(max_miou, diagonal=1).max(0)[0]
                sorted_indices_stuff = sorted_indices_stuff[max_miou < 0.6]
            
            newly_indices = torch.cat([sorted_indices_thing, sorted_indices_stuff])
            # slower speed but higher performance, because too much entities needed to generate pseudo prompts
            # newly_indices = newly_indices[nms_scores[newly_indices].sort(descending=True)[1][:self.test_topk_per_image]]
            # faster speed but slightly lower performance
            newly_indices = newly_indices[nms_scores[newly_indices] > self.apply_cls_thres]
        else:
            targets_per_video = targets[0]
            # find newly entities in the subsequent frames
            gt_logits = targets_per_video['logits']
            gt_masks = targets_per_video['masks']
            gt_mask_logits = targets_per_video['mask_logits']
            gt_boxes = targets_per_video['boxes']  # N_gt, T_prev, 4
            gt_embds = targets_per_video['embds']  # N_gt, T_prev, C
            gt_occurrence = targets_per_video['occurrence']
            gt_mask_quality_scores = targets_per_video["mask_quality_scores"]

            tgt_embds = gt_embds[:, -3:]
            if self.use_quasi_track:
                sim = torch.einsum('ntc,mfc->nmtf', tgt_embds, pred_embds).flatten(2)  # N_gt, N_pred, K
                sim_bi = (sim.softmax(1) + sim.softmax(0)).mean(-1) / 2. 
                sim_bi[sim_bi < self.detect_newly_object_threshold] = 0  # Important!!
                indices = linear_sum_assignment((1 - sim_bi).cpu())
                matched_sim = sim_bi[indices]
            else:
                indices, matched_sim = match_from_learnable_embds(
                    tgt_embds, pred_embds, return_similarity=True, return_src_indices=True, 
                    use_norm=False, thresh=self.detect_newly_object_threshold
                )
                
            above_sim = matched_sim > self.detect_newly_object_threshold
            matched_tgt_indices = torch.as_tensor(indices[0], device=matched_sim.device)[above_sim]
            matched_pred_indices = torch.as_tensor(indices[1], device=matched_sim.device)[above_sim]

            # update the detected entities stored in the memory pool
            matched_masks = F.interpolate(
                pred_masks[matched_pred_indices], interim_size, mode='bilinear', align_corners=False
            )
            nonblank_masks = matched_masks.flatten(-2).gt(0.).any(-1)
            gt_mask_logits[matched_tgt_indices, -num_frames:] += matched_masks.clone() 
            gt_occurrence[matched_tgt_indices, -num_frames:] += nonblank_masks.float()

            # !!Important: Must update GT embds from learnable queries, mismatch feature spaces yet
            gt_logits[matched_tgt_indices, -1] = 0.5 * (gt_logits[matched_tgt_indices, -1] + pred_logits[matched_pred_indices])
            nonblank_embds = (gt_embds[matched_tgt_indices, -1] != 0).any(-1)
            gt_embds[matched_tgt_indices, -1] = \
                (gt_embds[matched_tgt_indices, -1] + pred_embds[matched_pred_indices].mean(1)) / (nonblank_embds[..., None] + 1.)
            gt_mask_quality_scores[matched_tgt_indices] += mask_quality_scores[matched_pred_indices]

            targets_per_video['logits'] = gt_logits
            targets_per_video['embds'] = gt_embds
            targets_per_video['mask_logits'] = gt_mask_logits
            targets_per_video['masks'] = gt_mask_logits.gt(0.).float()
            targets_per_video['occurrence'] = gt_occurrence
            targets_per_video["mask_quality_scores"] = gt_mask_quality_scores

            # detect newly entities
            newly_indices = []
            gt_mask_logits_ds = F.interpolate(
                gt_mask_logits[:, -num_frames:], pred_masks.shape[-2:], mode='bilinear', align_corners=False
            )
            for idx in range(pred_embds.shape[0]):
                s_, l_ = pred_logits[idx].max(-1)
                if idx not in matched_pred_indices and s_ > 2*self.apply_cls_thres:
                    miou = batched_mask_iou(pred_masks[idx][:, None].gt(0.), gt_mask_logits_ds.transpose(0, 1).gt(0.))
                    if miou.nelement() and miou.max() < 0.5:
                        newly_indices.append(idx)
        
        out_learn['pred_logits'] = pred_logits[newly_indices]
        out_learn['pred_masks'] = pred_masks[newly_indices]
        out_learn['pred_embds'] = pred_embds[newly_indices]
        out_learn['pred_boxes'] = pred_boxes[newly_indices]
        out_learn['mask_quality_scores'] = mask_quality_scores[newly_indices]
    
    def write_newly_entities_into_annotations_per_clip(self, first_frame_idx, out, targets, interim_size):
        """
        Write annotated masks for these objects that appear in the first frame into the annotation Dict
        Args:
            out: A dict to store the masks, boxes of newly entites
            targets: A list with [Dict, Dict, ..], which stores the annotated masks of target-objects
            first_frame_idx: the indix of the first frame in the current processing clip
        """
        pred_logits = out['pred_logits'].unsqueeze(1) # Q_newlyx1xK
        pred_masks = out['pred_masks']   # Q_newlyxTxHxW
        pred_embds = out['pred_embds']   # Q_newlyxTxC
        pred_embds = torch.mean(pred_embds, dim=1, keepdim=True)
        pred_boxes = out['pred_boxes']   # Q_newlyxTx4
        mask_quality_scores = out['mask_quality_scores']   # Q_newlyxTx4
        _num_instance_newly = pred_masks.shape[0]

        num_frames = pred_masks.shape[1]
        
        first_appear_frame_idxs_newly = torch.ones(_num_instance_newly, dtype=torch.long, device=self.device) * first_frame_idx
        if _num_instance_newly == 0:
            pred_masks = torch.zeros((0,self.num_frames, interim_size[0], interim_size[1]), device=pred_masks.device)
        else:
            pred_masks = F.interpolate(
                pred_masks, interim_size, mode='bilinear', align_corners=False
            ) 
        pred_occurrence = torch.ones([pred_masks.shape[0], pred_masks.shape[1]], device=pred_masks.device)

        assert len(targets) == 1, "Only support the batch size is 1"
        targets_per_video = targets[0]
        if 'masks' not in targets_per_video:
            pred_ids = torch.arange(_num_instance_newly, device=self.device)
            targets_per_video.update({
                "logits": pred_logits, 
                "masks": pred_masks.gt(0.),  
                "mask_logits": pred_masks,  
                "boxes": pred_boxes,  
                "embds": pred_embds, 
                "ids":   pred_ids,
                "first_appear_frame_idxs": first_appear_frame_idxs_newly,
                "mask_quality_scores": mask_quality_scores,
                "occurrence": pred_occurrence,
            })
        
        else:
            if _num_instance_newly == 0:
                return 

            gt_logits = targets_per_video["logits"]  # N, num_frames_prev-num_frames+1, K
            gt_masks = targets_per_video['masks']    # N, num_frames+1, H, W
            gt_mask_logits = targets_per_video['mask_logits']  # N, num_frames+1, H, W
            gt_boxes = targets_per_video['boxes']    # N, num_frames_prev, 4
            gt_embds = targets_per_video['embds']    # N, num_frames_prev, C
            gt_ids   = targets_per_video["ids"]      # N
            gt_mask_quality_scores = targets_per_video["mask_quality_scores"]
            gt_occurrence = targets_per_video["occurrence"]
            T_prev = gt_boxes.shape[1]

            # zero-vector padding to keep consistency with annotations
            gt_logits_pad = torch.zeros([_num_instance_newly, gt_logits.shape[1]-pred_logits.shape[1], gt_logits.shape[-1]], dtype=torch.float, device=self.device)
            mask_shape = [_num_instance_newly, gt_masks.shape[1]-num_frames, interim_size[0], interim_size[1]]
            gt_masks_pad = torch.zeros(mask_shape, dtype=torch.float, device=self.device)
            gt_boxes_pad = torch.zeros([_num_instance_newly, gt_boxes.shape[1]-num_frames, 4], dtype=torch.float32, device=self.device)
            gt_embds_pad = torch.zeros([_num_instance_newly, gt_embds.shape[1]-pred_embds.shape[1], self.hidden_dim], dtype=torch.float32, device=self.device)
            gt_occurrence_pad = torch.zeros([_num_instance_newly, gt_occurrence.shape[1]-pred_occurrence.shape[1]], device=self.device)
            
            gt_logits_newly = torch.cat([gt_logits_pad, pred_logits], dim=1)
            gt_masks_newly = torch.cat([gt_masks_pad, pred_masks], dim=1)
            gt_boxes_newly = torch.cat([gt_boxes_pad, pred_boxes], dim=1)
            gt_embds_newly = torch.cat([gt_embds_pad, pred_embds], dim=1)
            gt_ids_newly = torch.arange(_num_instance_newly, device=self.device) + len(gt_ids)
            gt_occurrence_newly = torch.cat([gt_occurrence_pad, pred_occurrence], dim=1)
            
            gt_logits = torch.cat([gt_logits, gt_logits_newly]) # N+N_newly, num_frames_prev-num_frames+1, 4
            gt_masks = torch.cat([gt_masks, gt_masks_newly.gt(0.)])      # N+N_newly, num_frames+1, H, W
            print(first_frame_idx, gt_mask_logits.shape, gt_masks_newly.shape, self.apply_cls_thres)
            gt_mask_logits = torch.cat([gt_mask_logits, gt_masks_newly]) # N+N_newly, num_frames+1, H, W
            gt_boxes = torch.cat([gt_boxes, gt_boxes_newly])    # N+N_newly, num_frames_prev, 4
            gt_embds = torch.cat([gt_embds, gt_embds_newly])    # N+N_newly, num_frames_prev, C
            gt_ids = torch.cat([gt_ids, gt_ids_newly])          # N+N_newly, 
            gt_occurrence = torch.cat([gt_occurrence, gt_occurrence_newly])
            first_appear_frame_idxs = torch.cat([
                targets_per_video["first_appear_frame_idxs"], first_appear_frame_idxs_newly
            ])
            
            gt_mask_quality_scores = torch.cat([gt_mask_quality_scores, mask_quality_scores])
            targets_per_video.update({
                "logits": gt_logits, 
                "masks": gt_masks,  
                "mask_logits": gt_mask_logits,
                "boxes": gt_boxes,  
                "embds": gt_embds, 
                "ids":   gt_ids,
                "first_appear_frame_idxs": first_appear_frame_idxs,
                "mask_quality_scores": gt_mask_quality_scores,
                "occurrence": gt_occurrence,
            })
            
            if "prompt_pe" in targets_per_video:
                # pad the corresponding prompt informations for parallel
                prompt_pe = targets_per_video["prompt_pe"]
                prompt_feats = targets_per_video["prompt_feats"]
                prompt_attn_masks = targets_per_video["prompt_attn_masks"]
                prompt_pe_pad = torch.zeros([_num_instance_newly, *prompt_pe.shape[1:]], device=self.device)
                prompt_feats_pad = torch.zeros([_num_instance_newly, *prompt_feats.shape[1:]], device=self.device)
                prompt_attn_masks_pad = torch.zeros(
                    [prompt_attn_masks.shape[0], prompt_attn_masks.shape[1], _num_instance_newly, prompt_attn_masks.shape[-1]], device=self.device
                ).bool()
                targets_per_video["prompt_pe"] = torch.cat([prompt_pe, prompt_pe_pad])
                targets_per_video["prompt_feats"] = torch.cat([prompt_feats, prompt_feats_pad])
                targets_per_video["prompt_attn_masks"] = torch.cat([prompt_attn_masks, prompt_attn_masks_pad], dim=-2)
    
    def pad_zero_annotations_for_next_clip(self, targets, stride):
        targets_per_video = targets[0]
        gt_logits = targets_per_video["logits"]  # N, num_frames_prev-num_frames+1, K
        gt_masks = targets_per_video['masks']    # N, num_frames+1, H, W
        gt_mask_logits = targets_per_video['mask_logits']    # N, num_frames+1, H, W
        # gt_masks = gt_masks[:, max(gt_masks.shape[1]-self.num_frames, 0):]  # N, num_frames, H, W, emeory friendly
        # gt_mask_logits = gt_mask_logits[:, max(gt_mask_logits.shape[1]-self.num_frames, 0):]  # N, num_frames, H, W, emeory friendly
        gt_boxes = targets_per_video['boxes']    # N, num_frames_prev, 4
        gt_embds = targets_per_video['embds']    # N, num_frames_prev, C
        gt_ids   = targets_per_video["ids"]      # N
        gt_occurrence = targets_per_video['occurrence']
        _num_instance, T_prev, _ = gt_embds.shape

        # zero-vector padding to keep consistency with annotations
        mask_shape = [_num_instance, stride, gt_masks.shape[-2], gt_masks.shape[-1]]
        gt_masks_pad = torch.zeros(mask_shape, dtype=torch.float, device=self.device)
        gt_boxes_pad = torch.zeros([_num_instance, stride, 4], dtype=torch.float32, device=self.device)
        gt_embds_pad = torch.mean(gt_embds[:, -3:], dim=1, keepdim=True).clone()
        gt_occurrence_pad = torch.zeros([_num_instance, stride], device=self.device)

        gt_logits = torch.cat([gt_logits, gt_logits[:, -1:].clone()], dim=1) # N, num_frames_prev-num_frames+2, 4
        gt_masks = torch.cat([gt_masks, gt_masks_pad], dim=1)                # N, num_frames+1, H, W
        gt_mask_logits = torch.cat([gt_mask_logits, gt_masks_pad], dim=1)    # N, num_frames+1, H, W
        gt_boxes = torch.cat([gt_boxes, gt_boxes_pad], dim=1)                # N, num_frames_prev+1, 4
        gt_embds = torch.cat([gt_embds, gt_embds_pad], dim=1)                # N, num_frames_prev+1, C
        gt_occurrence = torch.cat([gt_occurrence, gt_occurrence_pad], dim=1)
            
        targets_per_video.update({
            "logits": gt_logits, 
            "masks": gt_masks, 
            "mask_logits": gt_mask_logits,  
            "boxes": gt_boxes,  
            "embds": gt_embds, 
            "occurrence": gt_occurrence,
        })

    def save_results_vis(self, first_frame_idx, targets, interim_size, image_size, out_size, is_last):
        targets_per_video = targets[0]
        if "masks" not in targets_per_video:
            return []  # no entity is detected
        
        frame_id_start = min(first_frame_idx + self.num_frames, targets_per_video['video_len']) \
            - targets_per_video["mask_logits"].shape[1] 

        mask_quality_scores = targets_per_video["mask_quality_scores"] 
        obj_ids = targets_per_video["ids"]            # cQ
        scores = targets_per_video["logits"].mean(1)  # cQ, K
        masks = targets_per_video["mask_logits"]
        occurence = targets_per_video["occurrence"]
        if not is_last:
            masks = masks[:, :self.num_frames_window_output] # cQ, W, H, W 
            occurence = occurence[:, :self.num_frames_window_output]
        masks = masks / occurence[..., None, None].clamp(min=1)
        
        masks = masks[:, :, : image_size[0], : image_size[1]]
        masks = retry_if_cuda_oom(F.interpolate)(
            masks.float(),
            size=out_size,
            mode="bilinear",
            align_corners=False
        ) 
        masks = (masks > 0.).cpu()
        scores = scores.cpu()

        results_list = []
        for i, (obj_id, s, mask) in enumerate(zip(obj_ids, scores, masks)):
            segms = [
                mask_util.encode(np.array(m[:, :, None], order="F", dtype="uint8"))[0]
                for m in mask.cpu()
            ]
            for rle in segms:
                rle["counts"] = rle["counts"].decode("utf-8")

            res = {
                "obj_id": int(obj_id),
                "score": s,
                "segmentations": segms,
                "frame_id_start": frame_id_start
            }
            if is_last:
                res['mask_quality_score'] = mask_quality_scores[i] / (int(mask_quality_scores.max()) + 1)
            results_list.append(res)

        return results_list
    
    def save_results_vps(
        self, first_frame_idx, targets, interim_size, image_size, out_size, is_last
    ):  
        targets_per_video = targets[0]
        gt_logits = targets_per_video["logits"]
        cur_masks = targets_per_video["mask_logits"]
        cur_occurrence = targets_per_video["occurrence"]
        cur_ids = targets_per_video["ids"]    # cQ
        gt_mask_quality_scores = targets_per_video["mask_quality_scores"]

        if not is_last:
            cur_masks = cur_masks[:, :self.num_frames_window_output] # cQ, W, H, W 
            cur_occurrence = cur_occurrence[:, :self.num_frames_window_output]

        cur_masks = cur_masks[:, :, : image_size[0], : image_size[1]]
        cur_masks = retry_if_cuda_oom(F.interpolate)(
            cur_masks.float(),
            size=out_size,
            mode="bilinear",
            align_corners=False
        ) 
        # cur_masks = cur_masks / cur_occurrence[..., None, None].clamp(min=1)

        if "stuff_memory_list" not in targets_per_video:
            targets_per_video["thing_memory_list"] = {}
            targets_per_video["stuff_memory_list"] = {}
        thing_memory_list = targets_per_video["thing_memory_list"]
        stuff_memory_list = targets_per_video["stuff_memory_list"]
        thing_stuff_segment_ids = list(thing_memory_list.values()) + list(stuff_memory_list.values())
        thing_obj_ids = list(thing_memory_list.keys())

        pred_cls = gt_logits.mean(1)  # cQ, K
        cur_scores, cur_classes = pred_cls.max(-1)
        cur_classes = cur_classes + 1 # category labels start from 1
        mask_quality_scores = calculate_mask_quality_scores(cur_masks)
        cur_scores = cur_scores * mask_quality_scores
        for k, pred_class in enumerate(cur_classes):
            isthing = pred_class.item() in self.metadata.thing_dataset_id_to_contiguous_id.keys()
            if k not in thing_obj_ids and not isthing:  # give a priority to thing entities
                cur_scores[k] *= 0.75 

        # initial panoptic_seg and segments infos
        num_insts, _, h, w = cur_masks.shape
        panoptic_seg = torch.zeros((cur_masks.size(1), out_size[0], out_size[1]), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return panoptic_seg.cpu()
        else:
            assert cur_ids.min() == 0 and cur_ids.max() == len(cur_ids) -1

            # interpolation to original image size
            cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks
            cur_masks = cur_masks.sigmoid()

            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            is_bg = (cur_masks < 0.5).sum(0) == len(cur_masks)
            cur_mask_ids[is_bg] = -1
            del cur_prob_masks

            current_segment_id = max(thing_stuff_segment_ids) + 1 if len(thing_stuff_segment_ids) else 0
            for k in range(cur_classes.shape[0]):
                cur_masks_k = cur_masks[k]
                pred_class = int(cur_classes[k])
                obj_id = int(cur_ids[k])
                # category labels start from 1
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.keys()
                
                # filter out the unstable segmentation results
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks_k >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks_k >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    overlap_threshold = 0.5*self.overlap_threshold if int(obj_id) in thing_obj_ids else self.overlap_threshold
                    if isthing and mask_area / original_area < overlap_threshold:
                        continue
                    
                    # merge stuff regions
                    if not isthing:
                        if pred_class not in stuff_memory_list.keys():
                            stuff_memory_list[pred_class] = current_segment_id + 1
                            current_segment_id += 1
                        segment_id = stuff_memory_list[pred_class]
                    else:
                        if obj_id not in thing_memory_list.keys():
                            thing_memory_list[obj_id] = current_segment_id + 1
                            current_segment_id += 1
                        segment_id = thing_memory_list[obj_id]

                    panoptic_seg[mask] = segment_id
            
            del cur_masks

            return panoptic_seg.cpu()
    
    def vps_output_results(self, targets, panoptic_seg_list, out_size):
        targets_per_video = targets[0]
        gt_logits = targets_per_video["logits"]
        gt_classes = gt_logits.mean(1).max(-1)[1] + 1  # category labels start from 1
        thing_memory_dict = targets_per_video["thing_memory_list"]
        stuff_memory_dict = targets_per_video["stuff_memory_list"]
        
        segments_infos = []
        for obj_id, segment_id in thing_memory_dict.items():
            obj_class = int(gt_classes[obj_id])
            segments_infos.append(
                {
                    "id": segment_id,
                    "isthing": obj_class in self.metadata.thing_dataset_id_to_contiguous_id, 
                    "category_id": obj_class, # category labels start from 1
                }
            )
        
        for stuff_class, segment_id in stuff_memory_dict.items():
            segments_infos.append(
                {
                    "id": segment_id,
                    "isthing": False,
                    "category_id": int(stuff_class), # category labels start from 1
                }
            )
        
        panoptic_seg = torch.cat(panoptic_seg_list, dim=0)
        return {
            "image_size": out_size,
            "pred_masks": panoptic_seg.cpu(),
            "segments_infos": segments_infos,
            "task": "vps",
        }

    def save_results_vss(
        self, first_frame_idx, output, interim_size, image_size, out_size, is_last, stride
    ):
        pred_logits = output["pred_logits"]  # cQ, K
        pred_masks = output["pred_masks"]
        if not is_last:
            pred_masks = pred_masks[:, :stride]  # cQ, Ts, H, W 
        
        pred_masks = retry_if_cuda_oom(F.interpolate)(
            pred_masks,
            size=interim_size,
            mode="bilinear",
            align_corners=False
        )
        pred_masks = pred_masks[:, :, : image_size[0], : image_size[1]]

        pred_masks = retry_if_cuda_oom(F.interpolate)(
            pred_masks.float(),
            size=out_size,
            mode="nearest",
        )
        
        mask_quality = calculate_mask_quality_scores(pred_masks)
        pred_logits = pred_logits * mask_quality.view(-1, 1)
        pred_masks = pred_masks.sigmoid()

        semseg = torch.einsum("qc,qthw->cthw", pred_logits, pred_masks)
        sem_mask = semseg.argmax(0)
        return sem_mask.cpu()
    
    def vss_output_results(self, targets, sem_mask_list, out_size):
        sem_mask = torch.cat(sem_mask_list, dim=0)
        return {
                "image_size": out_size,
                "pred_masks": sem_mask.cpu(),
                "task": "vss",
            }
    
    def visualize_results_vis(self, first_frame_idx, batched_inputs, targets, image_size, out_size, is_last):
        # batch size is 1 here
        targets_per_video = targets[0]
        file_names = batched_inputs[0]['file_names'] 
        video_len = len(file_names)
        video_id = file_names[0].split('/')[-2]
        

        save_dir = os.path.join(self.output_dir, 'inference/vis_'+targets_per_video['dataset_name'], video_id)
        os.makedirs(save_dir, exist_ok=True)

        first_frame_idx = min(first_frame_idx + self.num_frames, video_len) - targets_per_video['mask_logits'].shape[1]
        
        if is_last:
            pred_masks = targets_per_video['mask_logits']   # NTHW
            pred_scores = targets_per_video['logits'].mean(1)  # NK
        else:
            pred_masks = targets_per_video['mask_logits'][:, :self.num_frames_window_output] # NTHW
            pred_scores = targets_per_video['logits'][:, :self.num_frames_window_output].mean(1)  # NK
        pred_masks = pred_masks[:, :, :image_size[0], :image_size[1]]
        if pred_masks.nelement() == 0:
            print(
                f"Non objects are detected from frame {first_frame_idx} to {first_frame_idx+self.num_frames_window_output}:", 
                pred_masks.shape
            )
            return 
        
        ids = torch.as_tensor(targets_per_video["ids"], device=self.device)
        if image_size != out_size:
            pred_masks = F.interpolate(
                pred_masks.float(),
                out_size,
                mode='bilinear',
                align_corners=False
            ).gt(0.).float()
        
        pred_scores = pred_scores.cpu()
        pred_masks = pred_masks.cpu()

        for t, m in enumerate(pred_masks.transpose(0, 1)):
            file_name = file_names[first_frame_idx + t].split('/')[-1]
            img = batched_inputs[0]["image"][first_frame_idx + t]
            img = F.interpolate(
                img[None].float(),
                out_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0).long()
            img = img.permute(1, 2, 0).cpu().to(torch.uint8)
            visualizer = VisualizerFrame(
                img, metadata=self.metadata, 
                class_names=None
            )
            
            scores, classes = pred_scores.max(-1)
            is_low_score = scores < 0.02
            m[is_low_score] = 0

            results = Instances(out_size)
            results.pred_masks = m.cpu()
            results.scores = scores.tolist()
            results.pred_classes = classes   # +1? should start from 1

            save_path = '/'.join([save_dir, file_name.replace('.png', '.jpg')])
            VisImage = visualizer.draw_instance_predictions(results)
            VisImage.save(save_path)
            
        # is_last = True
        if is_last:
            # save all frames to a .avi video
            out = cv2.VideoWriter(
                '/'.join([save_dir, video_id + '.avi']),
                cv2.VideoWriter_fourcc(*'DIVX'),
                4,
                (out_size[1], out_size[0])
            )

            file_names = glob.glob('/'.join([save_dir, "*.jpg"]))
            file_names = sorted(file_names, key=lambda f: int(f.split("/")[-1].split('_')[-1].replace('.jpg', '')))
            for file_name in file_names:
                out.write(cv2.imread(file_name))
            out.release()
            print(f"save all frames with inst. seg. into {video_id}.avi")
    
    def visualize_results_vss(self, batched_inputs, sem_masks_dict, out_size, sub_task):
        file_names = batched_inputs[0]['file_names'] 
        dataset_name = file_names[0].split('/')[1]
        video_id = file_names[0].split('/')[-2]

        save_dir = os.path.join(self.output_dir, 'inference/vss_'+dataset_name, video_id)
        os.makedirs(save_dir, exist_ok=True)

        sem_masks = sem_masks_dict["pred_masks"]
        for t, sem_mask in enumerate(sem_masks):
            file_name = file_names[t].split('/')[-1]
            img = batched_inputs[0]["image"][t]
            img = F.interpolate(
                img[None].float(),
                out_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0).long()
            img = img.permute(1, 2, 0).cpu().to(torch.uint8)
            visualizer = VisualizerFrame(
                img, 
                metadata=self.metadata, 
                class_names=None
            )

            save_path = '/'.join([save_dir, file_name.replace('.png', '.ipg')])
            VisImage = visualizer.draw_sem_seg(sem_mask, alpha=0.6)
            VisImage.save(save_path)
        
        # save all frames to a .avi video
        out = cv2.VideoWriter(
            '/'.join([save_dir, video_id + '.avi']),
            cv2.VideoWriter_fourcc(*'DIVX'),
            4,
            (out_size[1], out_size[0])
        )

        file_names = glob.glob('/'.join([save_dir, "*.jpg"]))
        file_names = sorted(file_names, key=lambda f: int(f.split("/")[-1].split('_')[-1].replace('.jpg', '')))
        for file_name in file_names:
            out.write(cv2.imread(file_name))
        out.release()
        print(f"save all frames with sem. seg. into {video_id}.avi")
    
    def visualize_results_vps(self, batched_inputs, pan_seg_dict, out_size, sub_task):
        '''
        pan_seg_dict = {
            "image_size": out_size,
            "pred_masks": panoptic_seg.cpu(),  # V x H x W
            "segments_infos": segments_infos,
            "task": "vps",
            }
        '''
        file_names = batched_inputs[0]['file_names'] 
        dataset_name = file_names[0].split('/')[1]
        video_id = file_names[0].split('/')[-2]
        video_len = len(file_names)

        save_dir = os.path.join(self.output_dir, 'inference/vps_'+dataset_name, video_id)
        os.makedirs(save_dir, exist_ok=True)

        pan_segs = pan_seg_dict["pred_masks"]
        segm_infos = pan_seg_dict["segments_infos"]
        for segm_info in segm_infos:
            if segm_info['isthing']:
                segm_info['category_id'] = self.metadata.thing_dataset_id_to_contiguous_id[segm_info['category_id']]
            else:
                segm_info['category_id'] = self.metadata.stuff_dataset_id_to_contiguous_id[segm_info['category_id']]
        
        assert video_len == len(pan_segs), f'Mismatch length between predicted and GT masks: {len(pan_segs)} and {video_len}'
        for t, pan_seg in enumerate(pan_segs):
            file_name = file_names[t].split('/')[-1]
            img = batched_inputs[0]["image"][t]
            img = F.interpolate(
                img[None].float(),
                out_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0).long()
            img = img.permute(1, 2, 0).cpu().to(torch.uint8)
            visualizer = VisualizerFrame(
                img, 
                metadata=self.metadata, 
                class_names=None
            )

            save_path = '/'.join([save_dir, file_name.replace('.png', '.ipg')])
            VisImage = visualizer.draw_panoptic_seg(pan_seg, segm_infos, alpha=0.8)
            VisImage.save(save_path)
        
        # save all frames to a .avi video
        out = cv2.VideoWriter(
            '/'.join([save_dir, video_id + '.avi']),
            cv2.VideoWriter_fourcc(*'DIVX'),
            4,
            (out_size[1], out_size[0])
        )

        file_names = glob.glob('/'.join([save_dir, "*.jpg"]))
        file_names = sorted(
            file_names, 
            key=lambda f: int(f.split("/")[-1].split('_')[-1].replace('.jpg', ''))
        )
        for file_name in file_names:
            out.write(cv2.imread(file_name))
        out.release()
        print(f"save all frames with pan. seg. into {video_id}.avi")

    def plot_query_embds_per_video(self, targets):
        for targets_per_video in targets:
            video_name = targets_per_video["file_names"][0].split('/')[-2]

            pred_embds = targets_per_video['embds']  # N_pred, T_prev, C
            pred_logits = targets_per_video['logits'] # N_pred, T_prev, K
            for i, (embds, logits) in enumerate(zip(pred_embds, pred_logits)):
                s, l = logits.mean(0).max(-1)
                if s < self.apply_cls_thres:
                    continue

                embds = embds / torch.norm(embds, dim=-1, keepdim=True)
                embds = embds.clamp(min=0.)
                embds = embds / torch.max(embds, dim=-1, keepdim=True)[0].clamp(min=1e-3)
                embds_np = embds.t().cpu().numpy()
                # embds_np = (embds_np * 255).astype(np.uint8)

                save_by_category = True
                if save_by_category:
                    output_dir = self.output_dir + 'query_embds/category/' + str(int(l))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_path = os.path.join(output_dir, video_name + '_' + str(i) + '.jpg')
                else:
                    output_dir = self.output_dir + 'query_embds/video/' + video_name
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_path = os.path.join(output_dir, str(i) + '.jpg')

                im = plt.imshow(embds_np, cmap='jet', vmin=0, vmax=1)
                cbar = plt.colorbar(im, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
                plt.title(str(int(l))+' + %.2f' % s)
                plt.savefig(output_path, dpi=300,bbox_inches='tight')
                plt.clf()