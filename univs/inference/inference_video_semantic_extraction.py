import os
import cv2
import glob
import copy
import math
import json
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


class InferenceVideoSemanticExtraction(nn.Module):
    """
    Extract semantic map & learnable object queries
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
        # semantic extraction
        semantic_extraction_compression_ratio: int = 8,
        semantic_extraction_compression_ratio_temporal: int=1,
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
        self.num_frames_window_test = 2*num_frames
        self.semantic_extraction_compression_ratio = semantic_extraction_compression_ratio
        self.semantic_extraction_compression_ratio_temporal = semantic_extraction_compression_ratio_temporal
       
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
            # semantic extraction parameters
            "semantic_extraction_compression_ratio": cfg.MODEL.UniVS.TEST.SEMANTIC_EXTRACTION.COMPRESSION_RATIO,
            "semantic_extraction_compression_ratio_temporal": cfg.MODEL.UniVS.TEST.SEMANTIC_EXTRACTION.COMPRESSION_RATIO_TEMPORAL,
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
        
        return self.inference_video(model, batched_inputs, images_norm, targets)
        
    def inference_video(self, model, batched_inputs, images, targets):
        images_tensor = images.tensor
        video_len = int(batched_inputs[0]["video_len"])
        
        # masks size
        interim_size = images_tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])
        out_size = (out_height, out_width)

        obj_tokens_video = []
        compression_mask_features_video = []

        is_last = False
        start_idx_window, end_idx_window = 0, 0

        stride = self.num_frames 
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

            obj_tokens = out["pred_embds"]        # T, N_obj_tokens, C
            mask_features = out["mask_features"]  # T, C, H, W
            mask_features = F.interpolate(
                mask_features, 
                size=interim_size, 
                mode="bilinear", 
                align_corners=False
            )
            mask_features = mask_features[..., :image_size[0], :image_size[1]]
            
            s_itv = self.semantic_extraction_compression_ratio
            compression_height = int(out_height / s_itv)
            compression_width = int(out_width / s_itv)
            compression_mask_features = F.interpolate(
                mask_features,
                size=(compression_height, compression_width), 
                mode="nearest", 
            )
            
            obj_tokens_video.append(obj_tokens)
            compression_mask_features_video.append(compression_mask_features)
        
        obj_tokens_video = torch.cat(obj_tokens_video)  # T, C, N_obj_tokens
        compression_mask_features_video = torch.cat(compression_mask_features_video)  # T, C, H/32, W/32
        assert video_len == obj_tokens_video.shape[0]

        t_itv = self.semantic_extraction_compression_ratio_temporal
        # obj_tokens_video = obj_tokens_video[::t_itv]
        # compression_mask_features_video = compression_mask_features_video[::t_itv]
        # print(obj_tokens_video.shape, compression_mask_features_video.shape)

        # datasets/internvid/raw/InternVId-FLT_1/---3UsVESJA_00:03:31.638_00:03:41.638.mp4/
        out_dir = '/'.join(targets[0]['file_names'][0].split('/')[:-1])
        out_dir = out_dir.replace('raw', 'semantic_extraction')
        out_file_obj_tokens = out_dir.replace('.mp4', f"_obj_tokens_{s_itv}_{t_itv}.pt")
        out_file_compression_mask_features = out_dir.replace('.mp4', f"_compression_mask_features_{s_itv}_{t_itv}.pt")
        
        # Save tensor to a compressed file
        torch.save(obj_tokens_video[::t_itv], out_file_obj_tokens) #, _use_new_zipfile_serialization=True)
        torch.save(compression_mask_features_video[::t_itv], out_file_compression_mask_features) #, _use_new_zipfile_serialization=True)