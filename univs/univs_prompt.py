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
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
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
    InferenceImageGenericSegmentation,
    InferenceVideoVIS,
    InferenceVideoVISFast,
    InferenceVideoVPS,
    InferenceVideoVOS,
    InferenceVideoEntity,
    )
from univs.utils.comm import convert_mask_to_box, calculate_mask_quality_scores

from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info
from .prepare_targets import PrepareTargets


def convert_box_to_mask(outputs_box: torch.Tensor, h: int, w: int):
    box_normalizer = torch.as_tensor([w, h, w, h], dtype=outputs_box.dtype,
                                     device=outputs_box.device).reshape(1, 1, -1)
    outputs_box_wonorm = outputs_box * box_normalizer  # B, Q, 4
    outputs_box_wonorm = torch.cat([outputs_box_wonorm[..., :2].floor(),
                                    outputs_box_wonorm[..., 2:].ceil()], dim=-1)
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=outputs_box.device),
                                    torch.arange(w, device=outputs_box.device))  # H, W
    grid_y = grid_y.reshape(1, 1, h, w)
    grid_x = grid_x.reshape(1, 1, h, w)

    # repeat operation will greatly expand the computational graph
    gt_x1 = grid_x > outputs_box_wonorm[..., 0, None, None]
    lt_x2 = grid_x <= outputs_box_wonorm[..., 2, None, None]
    gt_y1 = grid_y > outputs_box_wonorm[..., 1, None, None]
    lt_y2 = grid_y <= outputs_box_wonorm[..., 3, None, None]
    cropped_box_mask = gt_x1 & lt_x2 & gt_y1 & lt_y2

    return cropped_box_mask


@META_ARCH_REGISTRY.register()
class UniVS_Prompt(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        lang_encoder: nn.Module or None,
        inference_img_generic_seg: nn.Module,
        inference_video_vis: nn.Module,
        inference_video_vis_fast: nn.Module,
        inference_video_vps: nn.Module,
        inference_video_vos: nn.Module,
        inference_video_entity: nn.Module,
        text_prompt_encoder,
        prepare_targets, 
        hidden_dim: int,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        stability_score_thresh: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        num_frames: int,
        num_classes: int,
        # boxvis
        gen_pseudo_mask: nn.Module,
        boxvis_enabled: bool,
        boxvis_ema_enabled: bool,
        boxvis_bvisd_enabled: bool,
        data_name: str,
        # inference
        video_unified_inference_enable: bool,
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
        num_frames_window_test: int=3,
        clip_stride: int=1,

    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            boxvis_enabled: if True, use only box-level annotation; otherwise pixel-wise annotations
            boxvis_ema_enabled: if True, use Teacher Net to produce high-quality pseudo masks
            boxvis_bvisd_enabled: if True, use box-supervised VIS dataset (BVISD), including
                pseudo video clip from COCO, videos from YTVIS21 and OVIS.
        """
        super().__init__()

        # Student Net
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        # self.lang_encoder = lang_encoder
        self.text_prompt_encoder = text_prompt_encoder
        self.inference_img_generic_seg = inference_img_generic_seg 
        self.inference_video_vis = inference_video_vis            # mdqe
        self.inference_video_vis_fast = inference_video_vis_fast  # minvis
        self.inference_video_vps = inference_video_vps
        self.inference_video_vos = inference_video_vos
        self.inference_video_entity = inference_video_entity
        self.prepare_targets = prepare_targets

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold
        self.stability_score_thresh = stability_score_thresh
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.num_classes = num_classes
        self.is_coco = data_name.startswith("coco")

        # boxvis
        if boxvis_enabled and boxvis_ema_enabled:
            # Teacher Net
            self.backbone_t = copy.deepcopy(backbone)
            self.sem_seg_head_t = copy.deepcopy(sem_seg_head)
            self.gen_pseudo_mask = gen_pseudo_mask
            self.backbone_t.requires_grad_(False)
            self.sem_seg_head_t.requires_grad_(False)
            self.ema_shadow_decay = 0.999

        self.boxvis_enabled = boxvis_enabled
        self.boxvis_ema_enabled = boxvis_ema_enabled
        self.boxvis_bvisd_enabled = boxvis_bvisd_enabled
        self.data_name = data_name
        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"

        # additional args reference
        self.video_unified_inference_enable = video_unified_inference_enable
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
        self.num_frames_window_test = num_frames_window_test
        self.clip_stride = clip_stride
        

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        lang_encoder = None
        if cfg.MODEL.UniVS.LANGUAGE_ENCODER_ENABLE and cfg.MODEL.UniVS.TEXT_PROMPT_ENCODER:
            lang_encoder = build_clip_language_encoder(cfg)
            # only eval
            for p in lang_encoder.parameters(): p.requires_grad = False
            lang_encoder.eval()
        text_prompt_encoder = None
        if cfg.MODEL.UniVS.TEXT_PROMPT_ENCODER:
            text_prompt_encoder = TextPromptEncoder(
                lang_encoder=lang_encoder,
                num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
            )
        prepare_targets = PrepareTargets(
            num_frames = cfg.INPUT.SAMPLING_FRAME_NUM,
            max_num_masks = cfg.MODEL.UniVS.NUM_POS_QUERIES,
            text_prompt_enable = cfg.MODEL.UniVS.TEXT_PROMPT_ENCODER,
            clip_class_embed_path = cfg.MODEL.UniVS.CLIP_CLASS_EMBED_PATH,
        )

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        reid_weight = cfg.MODEL.MASK_FORMER.REID_WEIGHT 
        proj_weight = dice_weight
        pair_weight = 1.
        if cfg.MODEL.BoxVIS.BoxVIS_ENABLED:
            dice_weight, mask_weight = 0.5*dice_weight, 0.5*mask_weight
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                           "loss_mask_proj": proj_weight, "loss_mask_pair": pair_weight}
        else:
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, 
                           "loss_reid": reid_weight, "loss_reid_aux": reid_weight, 
                           "loss_reid_l2p": reid_weight, "loss_reid_l2p_aux": reid_weight,
                           "loss_l2v_attn_weight": mask_weight, 
                           }

        class_weight_matcher = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT_MATCHER
        dice_weight_matcher = cfg.MODEL.MASK_FORMER.DICE_WEIGHT_MATCHER
        mask_weight_matcher = cfg.MODEL.MASK_FORMER.MASK_WEIGHT_MATCHER
        reid_weight_matcher = cfg.MODEL.MASK_FORMER.REID_WEIGHT_MATCHER
        # building criterion
        matcher = VideoHungarianMatcherUni(
            cost_class=class_weight_matcher,
            cost_mask=mask_weight_matcher,
            cost_dice=dice_weight_matcher,
            cost_proj=proj_weight,
            cost_pair=pair_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            boxvis_enabled=cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            boxvis_ema_enabled=cfg.MODEL.BoxVIS.EMA_ENABLED,
        )

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if cfg.MODEL.BoxVIS.EMA_ENABLED:
            gen_pseudo_mask = BoxVISTeacherSetPseudoMask(matcher)
        else:
            gen_pseudo_mask = None

        losses = ["labels", "masks", "reid"]
        criterion = VideoSetCriterionUni(
            cfg,
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )
        num_classes = sem_seg_head.num_classes

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "lang_encoder": lang_encoder,
            "text_prompt_encoder": text_prompt_encoder,
            "inference_img_generic_seg": InferenceImageGenericSegmentation(cfg),
            "inference_video_vis": InferenceVideoVIS(cfg),
            "inference_video_vis_fast": InferenceVideoVISFast(cfg),
            "inference_video_vps": InferenceVideoVPS(cfg),
            "inference_video_vos": InferenceVideoVOS(cfg),
            "inference_video_entity": InferenceVideoEntity(cfg),
            "prepare_targets": prepare_targets,
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "stability_score_thresh": cfg.MODEL.MASK_FORMER.TEST.STABILITY_SCORE_THRESH,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": num_classes,
            # boxvis
            "gen_pseudo_mask": gen_pseudo_mask,
            'boxvis_enabled': cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            "boxvis_ema_enabled": cfg.MODEL.BoxVIS.EMA_ENABLED,
            "boxvis_bvisd_enabled": cfg.MODEL.BoxVIS.BVISD_ENABLED,
            "data_name": cfg.DATASETS.TEST[0],
            # inference
            "video_unified_inference_enable": cfg.MODEL.UniVS.TEST.VIDEO_UNIFIED_INFERENCE_ENABLE,
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
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
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
        if not self.training:
            return self.forward_inference(batched_inputs)
        
        if self.boxvis_ema_enabled:
            # ---------------- prepare EMA for Teacher net ---------------------
            backbone_shadow, sem_seg_head_shadow = {}, {}
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    backbone_shadow[name] = param.data.clone().detach()

            for name, param in self.sem_seg_head.named_parameters():
                if param.requires_grad:
                    sem_seg_head_shadow[name] = param.data.clone().detach()

            w_shadow = 1.0 - self.ema_shadow_decay
            # apply weighted weights to the teacher net
            for name, param in self.backbone_t.named_parameters():
                if name in backbone_shadow:
                    param.data = w_shadow * backbone_shadow[name] + (1-w_shadow) * param.data
            for name, param in self.sem_seg_head_t.named_parameters():
                if name in sem_seg_head_shadow:
                    param.data = w_shadow * sem_seg_head_shadow[name] + (1-w_shadow) * param.data
                    
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images_norm = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_norm = ImageList.from_tensors(images_norm, self.size_divisibility)
        images = ImageList.from_tensors(images, self.size_divisibility)

        # image_size = images_norm.image_sizes[0]
        # out_height = batched_inputs[0].get("height", image_size[0])  # raw image size before data augmentation
        # out_width = batched_inputs[0].get("width", image_size[1])
        # out_size = (out_height, out_width)
        # print(image_size, images_norm.tensor.shape, out_size)

        targets = self.prepare_targets.process(batched_inputs, images, self.device, self.text_prompt_encoder)

        if self.boxvis_ema_enabled:
            # ------------------ Teacher Net -----------------------------
            features_t = self.backbone_t(images_norm.tensor)
            outputs_t = self.sem_seg_head_t(features_t, targets=targets)
            # generate pseudo masks via teacher outputs
            targets = self.gen_pseudo_mask(outputs_t, targets)

        features_s = self.backbone(images_norm.tensor)
        outputs_s = self.sem_seg_head(features_s, targets=targets)
        losses = self.criterion(outputs_s, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses
    
    def forward_inference(self, batched_inputs):
        dataset_name = batched_inputs[0]["dataset_name"]
        if dataset_name.startswith("coco") or dataset_name.startswith("ade20k"):
            # evaluation for images
            return self.inference_img_generic_seg.eval(self, batched_inputs)
    
        else:
            if dataset_name.startswith("sot") or dataset_name.startswith("rvos"):
                # evaluation for prompt-specified VS tasks
                return self.inference_video_vos.eval(self, batched_inputs)

            else:
                # evaluation for category-specified VS tasks
                if self.video_unified_inference_enable:
                    if dataset_name.startswith("ytvis") or dataset_name.startswith("ovis") \
                            or dataset_name.startswith("vipseg") or dataset_name.startswith("vspw"):
                        return self.inference_video_entity.eval(self, batched_inputs)
                    else:
                        raise ValueError(f"Not support to eval the dataset {dataset_name} yet")
                else:
                    if dataset_name.startswith("ytvis") or dataset_name.startswith("ovis"):
                        if self.tracker_type == 'mdqe':  # mdqe
                            return self.inference_video_vis.eval(self, batched_inputs)
                        else:  # minvis
                            return self.inference_video_vis_fast.eval(self, batched_inputs)
                    elif dataset_name.startswith("vipseg") or dataset_name.startswith("vpsw"):
                        return self.inference_video_vps.eval(self, batched_inputs)
                    else:
                        raise ValueError(f"Not support to eval the dataset {dataset_name} yet")