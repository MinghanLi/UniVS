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
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.utils.box_ops import box_xyxy_to_cxcywh
from univs import (
    VideoSetCriterion, 
    VideoHungarianMatcherUni,
    BoxVISTeacherSetPseudoMask,
    TextPromptEncoder,
    build_clip_language_encoder,
    PrepareTargets,
    InferenceImageGenericSegmentation,
    InferenceVideoVIS,
    InferenceVideoVPS,
    InferenceVideoVOS,
    )

from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info


@META_ARCH_REGISTRY.register()
class BoxVIS_VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        text_prompt_encoder: nn.Module,
        prepare_targets,
        criterion: nn.Module,
        inference_img_generic_seg: nn.Module,
        inference_video_vis: nn.Module,
        inference_video_vps: nn.Module,
        inference_video_vos: nn.Module,
        hidden_dim: int,
        num_queries: int,
        metadata,
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
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
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
        self.text_prompt_encoder = text_prompt_encoder
        self.prepare_targets = prepare_targets
        self.criterion = criterion
        self.inference_img_generic_seg = inference_img_generic_seg 
        self.inference_video_vis = inference_video_vis
        self.inference_video_vps = inference_video_vps
        self.inference_video_vos = inference_video_vos

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.num_classes = num_classes

        # boxvis
        self.boxvis_enabled = boxvis_enabled
        self.boxvis_ema_enabled = boxvis_ema_enabled
        self.boxvis_bvisd_enabled = boxvis_bvisd_enabled
        if boxvis_enabled and boxvis_ema_enabled:
            # Teacher Net
            self.backbone_t = copy.deepcopy(backbone)
            self.sem_seg_head_t = copy.deepcopy(sem_seg_head)
            self.gen_pseudo_mask = gen_pseudo_mask
            self.backbone_t.requires_grad_(False)
            self.sem_seg_head_t.requires_grad_(False)
            self.ema_shadow_decay = 0.999

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
                           "loss_reid": reid_weight, "loss_reid_aux": reid_weight}

        # building criterion
        matcher = VideoHungarianMatcherUni(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
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
        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            is_coco=cfg.DATASETS.TEST[0].startswith("coco"),
            # boxvis parameters
            boxvis_enabled=cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            boxvis_pairwise_enable=cfg.MODEL.BoxVIS.PAIRWISE_ENABLED,
            boxvis_pairwise_num_stpair=cfg.MODEL.BoxVIS.PAIRWISE_STPAIR_NUM,
            boxvis_pairwise_dilation=cfg.MODEL.BoxVIS.PAIRWISE_DILATION,
            boxvis_pairwise_color_thresh=cfg.MODEL.BoxVIS.PAIRWISE_COLOR_THRESH,
            boxvis_pairwise_corr_kernel_size=cfg.MODEL.BoxVIS.PAIRWISE_PATCH_KERNEL_SIZE,
            boxvis_pairwise_corr_stride=cfg.MODEL.BoxVIS.PAIRWISE_PATCH_STRIDE,
            boxvis_pairwise_corr_thresh=cfg.MODEL.BoxVIS.PAIRWISE_PATCH_THRESH,
            boxvis_ema_enabled=cfg.MODEL.BoxVIS.EMA_ENABLED,
            boxvis_pseudo_mask_score_thresh=cfg.MODEL.BoxVIS.PSEUDO_MASK_SCORE_THRESH,
            max_iters=cfg.SOLVER.MAX_ITER,
        )
        num_classes = sem_seg_head.num_classes

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "text_prompt_encoder": text_prompt_encoder,
            "prepare_targets": prepare_targets,
            "criterion": criterion,
            "inference_img_generic_seg": InferenceImageGenericSegmentation(cfg),
            "inference_video_vis": InferenceVideoVIS(cfg),
            "inference_video_vps": InferenceVideoVPS(cfg),
            "inference_video_vos": InferenceVideoVOS(cfg),
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
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
            dataset_name = batched_inputs[0]["dataset_name"]
            if dataset_name.startswith("coco") or dataset_name.startswith("ade20k"):
                return self.inference_img_generic_seg.eval(self, batched_inputs)
            elif dataset_name.startswith("ytvis") or dataset_name.startswith("ovis"):
                return self.inference_video_vis.eval(self, batched_inputs)
            elif dataset_name.startswith("vipseg") or dataset_name.startswith("vpsw"):
                return self.inference_video_vps.eval(self, batched_inputs)
            elif dataset_name.startswith("sot"):
                return self.inference_video_vos.eval(self, batched_inputs)
            else:
                raise ValueError(f"Not support evaluation on {dataset_name} dataset yet")

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images_norm = [(x - self.pixel_mean) / self.pixel_std for x in images]

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

        images_norm = ImageList.from_tensors(images_norm, self.size_divisibility)
        images = ImageList.from_tensors(images, self.size_divisibility)

        features_s = self.backbone(images_norm.tensor)
        outputs_s = self.sem_seg_head(features_s)
        targets = self.prepare_targets.process(batched_inputs, images, self.device)

        if self.boxvis_ema_enabled:
            # ------------------ Teacher Net -----------------------------
            features_t = self.backbone_t(images_norm.tensor)
            outputs_t = self.sem_seg_head_t(features_t)
            # generate pseudo masks via teacher outputs
            targets = self.gen_pseudo_mask(outputs_t, targets)

        losses = self.criterion(outputs_s, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses
