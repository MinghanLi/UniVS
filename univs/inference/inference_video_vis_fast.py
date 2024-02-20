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
    Clips, 
    FastOverTracker_DET,
    )
from univs.utils.comm import convert_mask_to_box, calculate_mask_quality_scores
from univs.prepare_targets import PrepareTargets

from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info
from .comm import match_from_learnable_embds, vis_clip_instances_to_coco_json_video


class InferenceVideoVISFast(nn.Module):
    """
    Class for inference on video instance segmentation task, 
    including frame-level tracker in MinVIS (NIPS2022) and MDQE (CVPR23)
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
        mdqe_tracker, 
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
        self.mdqe_tracker = mdqe_tracker
        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"
        self.num_max_inst_test = num_max_inst_test
        self.num_frames_window_test = max(num_frames_window_test, num_frames)
        self.clip_stride = clip_stride
        

    @classmethod
    def from_config(cls, cfg):
        mdqe_tracker = FastOverTracker_DET(cfg)
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
            "mdqe_tracker": mdqe_tracker,
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
        targets = model.prepare_targets.process_inference(batched_inputs, interim_size, self.device, model.text_prompt_encoder)

        dataset_name = batched_inputs[0]["dataset_name"]
        if dataset_name.startswith("ytvis") or dataset_name.startswith("ovis"):
            assert self.tracker_type == 'minvis', 'the type of tracker only supports minvis.'
            return self.inference_video_vis_minvis(model, batched_inputs, images_norm, targets=targets)
        else:
            raise ValueError(f'Do not support the model inference on {dataset_name}.')
    
    def inference_video_vis_minvis(self, model, batched_inputs, images, targets):
        images_tensor = images.tensor

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
            if i == 0:
                pred_scores = pred_logits.max(-1)[0]
                topk_idxs = torch.sort(pred_scores, descending=True)[1][:min(self.num_queries, 100)]
                out = {k: v[:, topk_idxs] for k, v in out.items() if not isinstance(v, list)}

            if self.merge_on_cpu:
                out = {k: v.cpu() for k, v in out.items() if not isinstance(v, list)}
            
            # remove the predicted semantic masks with thing categories
            pred_logits = out['pred_logits'][0, :self.num_queries].float() # QxK
            pred_masks = out['pred_masks'][0, :self.num_queries].float()   # QxTxHxW
            pred_embds = out['pred_embds'][0, :self.num_queries].float()   # QxTxC
            if i == 0:
                out_logits = [pred_logits]
                out_masks = [pred_masks]
                out_embds = [pred_embds.mean(1)]
            else:
                mem_embds = torch.stack(out_embds[-2:], dim=1)
                indices = match_from_learnable_embds(mem_embds, pred_embds)
                out_logits.append(pred_logits[indices, :])
                out_masks.append(pred_masks[indices, :, :, :])
                out_embds.append(pred_embds[indices, :].mean(1))

        q, n_t, h, w = out_masks[0].shape

        out_logits = sum(out_logits) / len(out_logits)
        dataset_name = batched_inputs[0]['dataset_name']
        assert dataset_name in combined_datasets_category_info
        num_classes, start_idx = combined_datasets_category_info[dataset_name]
        mask_scores = out_logits[..., start_idx:start_idx + num_classes]
        mask_scores = mask_scores.sigmoid()    # cos_sim with L2 norm

        out_masks_mean = []
        n_clips = len(out_masks)
        for v in range(n_clips+n_t-1):
            n_t_valid = min(v+1, n_t)
            m = []
            for t in range(n_t_valid):
                if v-t < n_clips:
                    m.append(out_masks[v-t][:, t])  # q, h, w
            out_masks_mean.append(torch.stack(m).mean(dim=0))  # q, h, w
            
        outputs = {}
        outputs['pred_masks'] = torch.stack(out_masks_mean, dim=1)  # t * [q h w] -> q t h w
        outputs['pred_scores'] = mask_scores                        # q k

        # masks size
        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])
        out_size = (out_height, out_width)

        return self.inference_video_vis_minvis_save_video(
            model, images, outputs, interim_size, image_size, out_size
        )
    
    def inference_video_vis_minvis_save_video(self, model, images, outputs, interim_size, image_size, out_size):
        mask_scores = outputs["pred_scores"]  # cQ, K
        mask_pred = outputs["pred_masks"]     # cQ, V, H, W 
        
        topk_idxs = mask_scores.max(-1)[0].sort(descending=True)[1][:self.test_topk_per_image]
        mask_scores = mask_scores[topk_idxs]
        mask_pred = mask_pred[topk_idxs]
        if self.zero_shot_inference:
            mask_scores = (mask_scores * 20).softmax(-1)  # towards to one-hot scores

        num_topk = min(self.test_topk_per_image, max(int((mask_scores > 2*(1./mask_scores.shape[-1])).sum()), 5))
        num_classes = mask_scores.shape[-1]
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_video, topk_indices = mask_scores.flatten(0, 1).topk(num_topk, sorted=False)
        labels_per_video = labels[topk_indices]
        topk_indices = torch.div(topk_indices, num_classes, rounding_mode='floor')

        mask_pred = mask_pred[topk_indices]
        mask_pred = retry_if_cuda_oom(F.interpolate)(
            mask_pred,
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        )  # cQ, t, H, W
        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]
        itv_t = max(int(mask_pred.shape[1] / 10.), 1)
        mask_quality_scores = calculate_mask_quality_scores(mask_pred[:, ::itv_t]).clamp(min=0.1)
        scores_per_video = scores_per_video * mask_quality_scores.to(scores_per_video.device)

        masks_per_video = []
        for m in mask_pred:
            # slower speed but memory friendly for long videos
            m = retry_if_cuda_oom(F.interpolate)(
                m.unsqueeze(0),
                size=out_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0) > 0.
            masks_per_video.append(m.cpu())

        scores_per_video = scores_per_video.tolist()
        labels_per_video = labels_per_video.tolist()

        processed_results = {
            "image_size": out_size,
            "pred_scores": scores_per_video,
            "pred_labels": labels_per_video,
            "pred_masks": masks_per_video,
        }

        return processed_results

    def inference_video_vis_mdqe(self, model, batched_inputs, images, targets):
        video_len = len(images.tensor)
        dataset_name = batched_inputs[0]['dataset_name']

        # OverTracker is memory-friendly, processing instance segmentation (long) clip by (long) clip
        # where the length of long clip is controlled by self.num_frames_window_test
        merge_device = "cpu" if self.merge_on_cpu else self.device
        
        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs[0].get("width", image_size[1])
        out_size = (out_height, out_width)
        self.mdqe_tracker.init_memory(
            num_insts=5, is_first=True, image_size=image_size, device=self.device
        )

        results_window_list = []
        start_idx_window, end_idx_window = 0, 0
        for i in range(video_len):
            targets[0]["frame_indices"] = torch.arange(i, i+self.num_frames)
            
            is_first_clip = i == 0
            is_last_clip = (i + self.num_frames) == (video_len - 1)
            if i + self.num_frames > video_len:
                break

            if i + self.num_frames > end_idx_window:
                start_idx_window, end_idx_window = i, i + self.num_frames_window_test
                frame_idx_window = range(start_idx_window, end_idx_window)
                features_window = model.backbone(images.tensor[start_idx_window:end_idx_window])

            features = {k: v[frame_idx_window.index(i):frame_idx_window.index(i)+self.num_frames]
                        for k, v in features_window.items()}
            outputs = model.sem_seg_head(features, targets=targets)
            del outputs['aux_outputs']
            if self.merge_on_cpu:
                outputs = {k: v.cpu() for k, v in outputs.items() if not isinstance(v, list)}

            i_map_to_0 = i % self.num_frames_window_test
            frame_idx_clip = list(range(i_map_to_0, i_map_to_0+self.num_frames))
            outputs_per_window = self.inference_video_vis_mdqe_per_clip(
                outputs, frame_idx_clip, dataset_name, interim_size, image_size, is_first_clip, is_last_clip
            )
            if outputs_per_window is not None:
                results_per_window = self.inference_video_vis_mdqe_save_window(
                    i, outputs_per_window, out_size, is_last_clip
                )
                results_window_list.append(results_per_window)
        
        return vis_clip_instances_to_coco_json_video(batched_inputs, results_window_list)

    def inference_video_vis_mdqe_per_clip(
        self, outputs, frame_idx_clip, dataset_name, interim_size, image_size, is_first_clip=False, is_last_clip=False, 
    ):
        q, n_t, h, w = outputs['pred_masks'][0, :self.num_queries].shape
        pred_cls_probs = outputs['pred_logits'][0, :self.num_queries].float()     # q k
        pred_masks = outputs['pred_masks'][0, :self.num_queries].float()       # q t h w
        pred_embds = outputs['pred_embds'][0, :self.num_queries].float()       # q t c 

        assert dataset_name in combined_datasets_category_info
        num_classes, start_idx = combined_datasets_category_info[dataset_name]
        pred_cls_probs = pred_cls_probs[..., start_idx:start_idx+num_classes]
        pred_cls_probs = pred_cls_probs.sigmoid()
        mask_quality_scores = calculate_mask_quality_scores(pred_masks)
        pred_cls_probs = pred_cls_probs * mask_quality_scores.reshape(-1, 1)

        scores, classes = pred_cls_probs.max(dim=-1)
        sorted_scores, sorted_idx = scores.sort(descending=True)
        valid_idx = sorted_idx[:max(1, int((sorted_scores > self.apply_cls_thres).sum()))]

        cos_sim_thresh_up = 0.9
        cos_sim_thresh_bottom = 0.75
        if not is_first_clip:
            mem_embds = self.mdqe_tracker.get_query_embds().unsqueeze(1)
            mem_indices, mem_scores = match_from_learnable_embds(mem_embds, pred_embds, return_similarity=True)
            mem_indices = [idx for idx, s in zip(mem_indices, mem_scores) if s > cos_sim_thresh_up]
            if len(mem_indices) > 0:
                sim_mem = torch.mm(
                    F.normalize(pred_embds[valid_idx].mean(1), dim=-1),
                    F.normalize(pred_embds[mem_indices].mean(1), dim=-1).t(),
                )
                valid_idx = valid_idx[sim_mem.max(-1)[0] < cos_sim_thresh_bottom]

        if len(valid_idx):
            enable_mask_iou = False
            if enable_mask_iou:
                # Non-maximum suppression (NMS) based on mask IoU
                m_soft = pred_masks[valid_idx].sigmoid().flatten(1)
                numerator = m_soft[None] * m_soft[:, None].gt(0.5)
                denominator = m_soft[None] + m_soft[:, None].gt(0.5) - numerator
                siou = numerator.sum(dim=-1) / denominator.sum(dim=-1)
                max_siou = torch.triu(siou, diagonal=1).max(dim=0)[0]
                valid_idx = valid_idx[max_siou < 0.6]
            else:
                sim = torch.mm(
                    F.normalize(pred_embds[valid_idx].mean(1), dim=-1),
                    F.normalize(pred_embds[valid_idx].mean(1), dim=-1).t(),
                ) 
                max_sim = torch.triu(sim, diagonal=1).max(dim=0)[0]
                valid_idx = valid_idx[max_sim < 0.65]

        valid_idx = valid_idx[:self.test_topk_per_image]
        if not is_first_clip:
            valid_idx = list(set(mem_indices + valid_idx.tolist()))
        
        pred_masks = pred_masks[valid_idx]
        pred_masks = retry_if_cuda_oom(F.interpolate)(
            pred_masks,
            size=interim_size,
            mode="bilinear",
            align_corners=False
        )
        pred_masks = pred_masks[:, :, :image_size[0], :image_size[1]]

        clip_results = Clips(image_size, frame_idx_clip)
        clip_results.scores = scores[valid_idx]
        clip_results.classes = classes[valid_idx]
        clip_results.cls_probs = pred_cls_probs[valid_idx]
        clip_results.mask_logits = pred_masks
        clip_results.query_embeds = pred_embds[valid_idx].mean(1)

        # update the current clip
        self.mdqe_tracker.update(clip_results)

        # Save output results clip by clip, which is memory-friendly. After inference of the video,
        # the instance masks of all clips will be directly merged into the .json file (mdqe/data/ytvis_eval.py).
        is_output = ((frame_idx_clip[0] + 1) % self.num_frames_window_test) == 0
        if is_last_clip or is_output:
            return self.mdqe_tracker.get_result(is_last_window=is_last_clip)
        else:
            return None

    def inference_video_vis_mdqe_save_window(
        self, cur_frame_idx, outputs_window, out_size, is_last_clip
    ):
        pred_masks = outputs_window["pred_masks"].cpu()              # cQ x T_w x H x W
        pred_scores = outputs_window["pred_cls_scores"].cpu()        # cQ x K
        pred_obj_ids = outputs_window["obj_ids"]                     # cQ, gt_ids for sot task

        pred_masks_list = []
        num_nonblank_masks_list = []
        for m in pred_masks:
            # slower speed but memory efficiently for long videos
            m = retry_if_cuda_oom(F.interpolate)(
                m.unsqueeze(0),
                size=out_size,
                mode="bilinear",
                align_corners=False
            ).squeeze(0) > 0.
            pred_masks_list.append(m.cpu())
            num_nonblank_masks_list.append((m.sum((-2, -1)) > 0).sum())

        results_per_window_list = []
        for obj_id, s, m, nonblank in zip(pred_obj_ids, pred_scores, pred_masks_list, num_nonblank_masks_list):
            segms = [
                mask_util.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
                for _mask in m
            ]
            for rle in segms:
                rle["counts"] = rle["counts"].decode("utf-8")

            frame_id_start = cur_frame_idx + 1 - len(segms) if not is_last_clip \
                else cur_frame_idx + self.num_frames - len(segms)
            res = {
                "obj_id": int(obj_id),
                "score": np.array(s),
                "segmentations": segms,
                "num_nonblank_masks": np.array(nonblank),
                "frame_id_start": frame_id_start
            }
            results_per_window_list.append(res)

        return results_per_window_list