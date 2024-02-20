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


class InferenceVideoVPS(nn.Module):
    """
    Class for inference on video panoptic segmentation task, 
    where we empoly a simple frame-level tracker in MinVIS (NIPS2022)
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
        self.test_topk_per_image = test_topk_per_image
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres
        self.window_inference = window_inference
        self.merge_on_cpu = merge_on_cpu
        
        # clip-by-clip tracking
        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"
        self.num_max_inst_test = num_max_inst_test
        self.num_frames_window_test = max(num_frames_window_test, num_frames)
        self.clip_stride = clip_stride
        
        self.change_to_720p = True

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
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        if self.LSJ_aug_enable_test:
            padding_constraints = {"size_divisibility": self.size_divisibility, "square_size": self.LSJ_aug_image_size}
            images = ImageList.from_tensors(images, padding_constraints=padding_constraints)
        else:
            images = ImageList.from_tensors(images, self.size_divisibility)

        interim_size = images.tensor.shape[-2:]
        targets = model.prepare_targets.process_inference(batched_inputs, interim_size, self.device, model.text_prompt_encoder)

        dataset_name = batched_inputs[0]["dataset_name"]
        if dataset_name.startswith("vipseg"):
            return self.inference_video_vps_online(model, batched_inputs, images, targets)
        else:
            raise ValueError(f'Not support to eval {dataset_name} during training yet.')
    
    def inference_video_vps_online(self, model, batched_inputs, images, targets):
        images_tensor = images.tensor

        # compared to tracker in MinVIS, this is more friendly for memory
        start_idx_window, end_idx_window = 0, 0
        for i in range(len(images_tensor)):
            targets[0]["frame_indices"] = torch.arange(i, i+self.num_frames)
            
            if i + self.num_frames > len(images_tensor):
                break

            if i + self.num_frames > end_idx_window:
                start_idx_window, end_idx_window = i, i + self.num_frames_window_test
                frame_idx_window = range(start_idx_window, end_idx_window)
                features_window = model.backbone(images_tensor[start_idx_window:end_idx_window])

            features = {k: v[frame_idx_window.index(i):frame_idx_window.index(i)+self.num_frames]
                        for k, v in features_window.items()}
            out = model.sem_seg_head(features, targets=targets)
            del out['aux_outputs']

            pred_logits = out['pred_logits'][0].sigmoid() 
            if self.stability_score_thresh > 0:
                scores_mask = calculate_mask_quality_scores(out['pred_masks'][0])
                pred_logits = pred_logits + scores_mask.view(-1,1)
            
            pred_scores = pred_logits.max(-1)[0]
            topk_idxs = torch.sort(pred_scores, descending=True)[1][:min(self.num_queries, 100)]
            if i == 0:
                out = {k: v[:, topk_idxs] for k, v in out.items() if not isinstance(v, list)}

            if self.merge_on_cpu:
                out = {k: v.cpu() for k, v in out.items()}
            
            pred_logits = out['pred_logits'][0].float() # QxK
            pred_masks = out['pred_masks'][0].float()   # QxtxHxW
            pred_embds = out['pred_embds'][0].float()   # QxtxC 
            pred_embds = pred_embds.mean(1)             # QxtxC -> Qxt

            # TODO: remove prompt queries directly
            pred_logits = pred_logits[:self.num_queries]
            pred_masks = pred_masks[:self.num_queries]
            pred_embds = pred_embds[:self.num_queries]
            
            if i == 0:
                out_logits = [pred_logits]
                out_masks = [pred_masks]
                out_embds = [pred_embds]
            else:
                mem_embds = torch.stack(out_embds[-2:]).mean(dim=0)
                indices = self.match_from_embds(mem_embds, pred_embds)
                out_logits.append(pred_logits[indices, :])
                out_masks.append(pred_masks[indices, :, :, :])
                out_embds.append(pred_embds[indices, :])
        
        q, n_t, h, w = out_masks[0].shape
        n_clips = len(out_masks)

        out_logits = sum(out_logits) / len(out_logits)
        out_masks_mean = []
        for v in range(n_clips+n_t-1):
            n_t_valid = min(v+1, n_t)
            m = []
            for t in range(n_t_valid):
                if v-t < n_clips:
                    m.append(out_masks[v-t][:, t])  # q, h, w
            out_masks_mean.append(torch.stack(m).mean(dim=0))  # q, h, w

        dataset_name = batched_inputs[0]['dataset_name']
        assert dataset_name in combined_datasets_category_info
        num_classes, start_idx = combined_datasets_category_info[dataset_name]
        out_logits = out_logits[..., start_idx:start_idx + num_classes]
        pred_cls = out_logits.sigmoid()                  # q k, cos_sim with L2 norm
        pred_masks = torch.stack(out_masks_mean, dim=1)  # t * [q h w] -> q t h w

         # upsample masks
        interim_size = images_tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])
        out_size = (out_height, out_width)
        if self.change_to_720p:
            out_size = (720, int(720 * out_width/out_height))

        return self.inference_video_vps_save_results(pred_cls, pred_masks, interim_size, image_size, out_size)

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))

        cost_embd = 1 - cos_sim
        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices
    
    def inference_video_vps_save_results(
        self, pred_cls, pred_masks, interim_size, img_size, out_size
    ):
        n, len_vid, h, w = pred_masks.shape
        scores, labels = pred_cls.max(-1)
        pred_id = torch.arange(len(scores), device=pred_cls.device)

        # filter out the predictions with low scores
        keep = scores > max(self.object_mask_threshold, scores.topk(k=self.test_topk_per_image)[0][-1])
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = pred_masks[keep]
        cur_ids = pred_id[keep]
        del pred_masks

        # initial panoptic_seg and segments infos
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((cur_masks.size(1), out_size[0], out_size[1]), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return {
                "image_size": out_size,
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }
        else:
            # interpolation to original image size
            t_itv = 10
            cur_masks = torch.cat([
                F.interpolate(
                    cur_masks[:, t:t+t_itv], size=interim_size, mode="bilinear", align_corners=False
                )[:, :, :img_size[0], :img_size[1]]
                for t in range(0, cur_masks.shape[1], t_itv)
            ], dim=1)
            mask_quality_scores = calculate_mask_quality_scores(cur_masks[:, ::5])  # temporal sparsity to save memory
            cur_scores = cur_scores + 0.5 * mask_quality_scores
            cur_masks = cur_masks.sigmoid()
            is_bg = (cur_masks < 0.5).sum(0) == len(cur_masks)
            cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks

            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            cur_mask_ids[is_bg] = -1
            del cur_prob_masks
            cur_mask_ids = F.interpolate(
                cur_mask_ids.float().unsqueeze(0), size=out_size, mode="nearest", 
            ).long().squeeze(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                # memory friendly
                cur_masks_k = F.interpolate(
                    cur_masks[k].unsqueeze(0), size=out_size, mode="bilinear", align_corners=False
                ).squeeze(0)

                pred_class = int(cur_classes[k]) + 1  # should start from 1
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.keys()
                
                # filter out the unstable segmentation results
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks_k >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks_k >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if pred_class in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[pred_class]
                            continue
                        else:
                            stuff_memory_list[pred_class] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_infos.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": pred_class,
                        }
                    )
                    out_ids.append(cur_ids[k])

            del cur_masks
            
            return {
                "image_size": out_size,
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }