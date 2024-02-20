import copy
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
# support color space pytorch, https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html#rgb_to_lab
from kornia import color
from scipy.optimize import linear_sum_assignment

import numpy as np
import pycocotools.mask as mask_util

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
    )
from univs.utils.comm import convert_mask_to_box, calculate_mask_quality_scores
from univs.prepare_targets import PrepareTargets

from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info
from .visualization import display_instance_masks


class InferenceImageGenericSegmentation(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        stability_score_thresh: float,
        metadata,
        size_divisibility: int,
        LSJ_aug_image_size: int,
        LSJ_aug_enable_test: bool,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        num_frames: int,
        data_name: str,
        # inference
        prompt_as_queries: bool,
        zero_shot_inference: bool,
        semantic_on: bool,
        instance_on: bool,
        panoptic_on: bool,
        disable_semantic_queries: bool,
        test_topk_per_image: int,
        tracker_type: str,
        window_inference: bool,
        is_multi_cls: bool,
        apply_cls_thres: float,
        merge_on_cpu: bool,
        # tracking
        num_max_inst_test: int,
        num_frames_window_test: int,
        clip_stride: int,

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
        self.disable_semantic_queries = disable_semantic_queries
        self.test_topk_per_image = test_topk_per_image
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres
        self.window_inference = window_inference
        self.merge_on_cpu = merge_on_cpu
        
        # clip-by-clip tracking
        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"
        self.num_max_inst_test = num_max_inst_test
        self.num_frames_window_test = num_frames_window_test
        self.clip_stride = clip_stride
        

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
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
            "data_name": cfg.DATASETS.TEST[0],
            # inference
            "prompt_as_queries": cfg.MODEL.UniVS.PROMPT_AS_QUERIES,
            "zero_shot_inference": cfg.MODEL.BoxVIS.TEST.ZERO_SHOT_INFERENCE,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "disable_semantic_queries": cfg.MODEL.UniVS.TEST.DISABLE_SEMANTIC_QUERIES,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "tracker_type": cfg.MODEL.BoxVIS.TEST.TRACKER_TYPE,
            "window_inference": cfg.MODEL.BoxVIS.TEST.WINDOW_INFERENCE,
            "is_multi_cls": cfg.MODEL.BoxVIS.TEST.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.BoxVIS.TEST.APPLY_CLS_THRES,
            "merge_on_cpu": cfg.MODEL.BoxVIS.TEST.MERGE_ON_CPU,
            # tracking
            "num_max_inst_test": cfg.MODEL.BoxVIS.TEST.NUM_MAX_INST,
            "num_frames_window_test": cfg.MODEL.BoxVIS.TEST.NUM_FRAMES_WINDOW,
            "clip_stride": cfg.MODEL.BoxVIS.TEST.CLIP_STRIDE,
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
        targets = model.prepare_targets.process_inference(
            batched_inputs, interim_size, self.device, model.text_prompt_encoder
        )

        dataset_name = batched_inputs[0]["dataset_name"]
        if dataset_name.startswith("coco") or dataset_name.startswith("ade20k"):
            return self.inference_image(model, batched_inputs, images_norm, targets)
        else:
            raise ValueError(f'do not support the model inference on {dataset_name}.')
    
    def inference_image(self, model, batched_inputs, images, targets):
        features = model.backbone(images.tensor)
        outputs = model.sem_seg_head(features, targets=targets)
        del outputs['aux_outputs']

        dataset_name = batched_inputs[0]['dataset_name']
        assert dataset_name in combined_datasets_category_info
        num_classes, start_idx = combined_datasets_category_info[dataset_name]
        mask_cls_results = outputs["pred_logits"][..., start_idx:start_idx + num_classes]
        mask_pred_results = outputs["pred_masks"][..., 0, :, :]
        mask_reid_results = outputs["pred_reid_logits"]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, mask_reid_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, mask_reid_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_cls_result = mask_cls_result.sigmoid()  
            scores_mask = calculate_mask_quality_scores(mask_pred_result)
            mask_cls_result = mask_cls_result * scores_mask.unsqueeze(-1)
            if self.stability_score_thresh > 0:
                keep = scores_mask > self.stability_score_thresh
                mask_cls_result = mask_cls_result[keep]
                mask_pred_result = mask_pred_result[keep]
           
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)
            else:
                mask_pred_result = mask_pred_result[:, :image_size[0], :image_size[1]]
            
            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    panoptic_mask = F.interpolate(
                        panoptic_r[0][None, None].float(), 
                        size=(height, width), 
                        mode="nearest", 
                    )[0, 0]
                    segment_ids = panoptic_mask.to(dtype=torch.int32).unique()
                    segments_info = [info for info in panoptic_r[1] if info["id"] in segment_ids]
                    panoptic_r = (panoptic_mask.to(dtype=torch.int32), segments_info)
                processed_results[-1]["panoptic_seg"] = panoptic_r
            
            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, (height, width))
                processed_results[-1]["instances"] = instance_r
                # display_instance_masks(batched_inputs[0], self.metadata, instance_r)

        return processed_results
    
    def semantic_inference(self, mask_cls, mask_pred):
        if self.prompt_as_queries and self.disable_semantic_queries:
            # remove thing categories from the predicted semantic masks 
            is_thing_idxs = self.metadata.thing_dataset_id_to_contiguous_id.values()
            mask_cls = mask_cls[self.num_queries:]
            mask_pred = mask_pred[self.num_queries:]

        use_topk = True
        if use_topk:
            keep_by_topk = torch.topk(mask_cls.max(-1)[0], k=200)[1]
            mask_cls = mask_cls[keep_by_topk]
            mask_pred = mask_pred[keep_by_topk]

        mask_pred = mask_pred.sigmoid()
        mask_cls = (mask_cls / 0.06).softmax(-1)  # !!! for open-voc settings
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        if self.prompt_as_queries:
            # remove the predicted semantic masks with thing categories
            is_thing_idxs = self.metadata.thing_dataset_id_to_contiguous_id.values()
            mask_cls = torch.stack([m for i, m in enumerate(mask_cls) 
                                    if i < self.num_queries or i-self.num_queries not in is_thing_idxs])
            mask_pred = torch.stack([s for i, s in enumerate(mask_pred) 
                                    if i < self.num_queries or i-self.num_queries not in is_thing_idxs])
        
        nms_enable = True
        if nms_enable:
            mask_cls, mask_pred, _ = self.postprocess_nms(mask_cls, mask_pred, biou_threshold=0.9)

        scores, labels = mask_cls.max(-1)
        mask_pred = mask_pred.sigmoid()
        
        num_classes = mask_cls.shape[-1]
        keep = (scores > self.object_mask_threshold)
        scores, labels = (mask_cls / 0.06).softmax(-1).max(-1)  # !!! for open-voc settings
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
    
        num_masks, h, w = cur_masks.shape
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if num_masks == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # NHW -> HW
            stuff_memory_list = {}
            nk = 0
            for k in range(num_masks):
                pred_class = cur_classes[k].item()
                isthing = int(pred_class) in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    nk += 1
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
           
            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, out_size):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        box_pred = convert_mask_to_box(mask_pred.gt(0)) 
        if self.prompt_as_queries:
            mask_cls = mask_cls[:self.num_queries]
            mask_pred = mask_pred[:self.num_queries]
            box_pred = box_pred[:self.num_queries]
        
        # if this is panoptic segmentation, we only keep the "thing" classes
        if len(self.metadata.thing_dataset_id_to_contiguous_id) != mask_cls.shape[-1]:
            # used in original Mask2Former
            labels = mask_cls.max(-1)[1]
            mask_cls = mask_cls[..., list(self.metadata.thing_dataset_id_to_contiguous_id.values())]

            keep = torch.zeros_like(labels).bool()
            for i, lab in enumerate(labels):
                keep[i] = int(lab) in self.metadata.thing_dataset_id_to_contiguous_id.values()
            if keep.sum() == 0:
                scores = mask_cls.max(-1)[0]
                keep = scores >= min(0.1, scores.max())

            mask_cls = mask_cls[keep]
            mask_pred = mask_pred[keep]
            box_pred = box_pred[keep]
        
        nms_enable = True
        if nms_enable:
            mask_cls, mask_pred, box_pred = self.postprocess_nms(mask_cls, mask_pred, box_pred)

        num_classes = mask_cls.shape[-1]
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(mask_cls.shape[0], 1).flatten(0, 1)
    
        k = min(self.test_topk_per_image, mask_cls.nelement())
        scores_per_image, topk_indices = mask_cls.flatten(0, 1).topk(k, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, num_classes, rounding_mode='floor').long()
        mask_pred = mask_pred[topk_indices].detach()
        box_pred = box_pred[topk_indices].detach()

        if image_size != out_size:
            mask_pred = F.interpolate(
                mask_pred[None], size=out_size, mode="bilinear", align_corners=False
            )[0]
            box_pred = convert_mask_to_box(mask_pred.gt(0)) 

        result = Instances(out_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(box_pred)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        result.scores = scores_per_image 
        result.pred_classes = labels_per_image

        return result
    
    def postprocess_nms(self, scores, mask_pred, box_pred=None, biou_threshold=0.85):
        if box_pred is None:
            box_pred = convert_mask_to_box(mask_pred.gt(0.))
        scores_nms, labels_nms = scores.max(-1)
        keep_by_nms = batched_nms(
            box_pred.float(),
            scores_nms,
            labels_nms,  # categories
            iou_threshold=biou_threshold,
        )
        scores = scores[keep_by_nms]
        mask_pred = mask_pred[keep_by_nms]
        box_pred = box_pred[keep_by_nms]

        return scores, mask_pred, box_pred