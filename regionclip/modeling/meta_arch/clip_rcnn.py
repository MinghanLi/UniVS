# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling import META_ARCH_REGISTRY

from detectron2.modeling import (
    Backbone,
    build_roi_heads,
)

from regionclip.modeling.backbone.clip_backbone import build_clip_resnet_backbone
from videosam.modeling.language.TextEncoder import build_clip_language_encoder

__all__ = ["CLIPFastRCNN"]


class CLIPFastRCNN(nn.Module):
    """
    Fast R-CNN style where the cropping is conducted on feature maps instead of raw images.
    It contains the following two components:
    1. Localization branch: pretrained backbone+RPN or equivalent modules, and is able to output object proposals
    2. Recognition branch: is able to recognize zero-shot regions
    """

    def __init__(
            self,
            *,
            backbone: Backbone,
            language_encoder: nn.Module,
            roi_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            clip_crop_region_type: str = 'GT',
            use_clip_attpool: False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.lang_encoder = language_encoder
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        # input format, pixel mean and std for offline modules
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0:  # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False

        self.clip_crop_region_type = clip_crop_region_type
        # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool
        self.use_clip_attpool = use_clip_attpool

    @property
    def device(self):
        return self.pixel_mean.device

    def inference(
            self,
            batched_inputs,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (Dict[str, list]) or List[Dict, Dict, ...]: same as in :meth:`forward`

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert self.clip_crop_region_type == "GT"

        # recognition branch: get 2D feature maps using the backbone of recognition branch
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert isinstance(batched_inputs, (List, Dict))
        # bounding boxes come from ground-truth annotations
        if isinstance(batched_inputs, Dict):
            # Multiple frames from a video clip => List of gt boxes
            proposals = [
                b_input._fields['gt_boxes'].to(self.device) for b_input in batched_inputs["instances"]
            ]

        else:
            # A mini-batch of single images => List of gt boxes
            proposals = [
                b_input["instances"]._fields['gt_boxes'].to(self.device)
                for r_i, b_input in enumerate(batched_inputs)
            ]

        # Given the proposals, crop region features from 2D image features and classify the regions
        # use att_pool from CLIP to match dimension
        attnpool = self.backbone.attnpool if self.use_clip_attpool else None
        # we employ log_softmax on class scores for torch.nn.KLDivLoss
        # so cls_probs = cls_log_probs_list.exp()
        cls_log_probs_list = self.roi_heads(features, proposals, res5=self.backbone.layer4, attnpool=attnpool)

        if isinstance(batched_inputs, Dict):
            for cls_log_probs, instance_per_image in zip(cls_log_probs_list, batched_inputs["instances"]):
                instance_per_image.pseudo_class_log_probs = cls_log_probs
        else:
            for cls_log_probs, input_per_image in zip(cls_log_probs_list, batched_inputs):
                input_per_image["instances"].pseudo_class_log_probs = cls_log_probs

        return batched_inputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border.
              So FPN is consistent with C4 and GT boxes.
        """
        if isinstance(batched_inputs, List):
            images = [x["image"].to(self.device) for x in batched_inputs]
        else:
            images = [x.to(self.device) for x in batched_inputs["image"]]

        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


def build_CLIPFastRCNN(cfg):
    assert cfg.MODEL.CLIP.CROP_REGION_TYPE == "GT"

    backbone = build_clip_resnet_backbone(cfg)
    # build language encoder and extract concept embeddings
    language_encoder = build_clip_language_encoder(cfg)

    roi_heads = build_roi_heads(cfg, backbone.output_shape())

    return CLIPFastRCNN(
        backbone=backbone,
        language_encoder=language_encoder,
        roi_heads=roi_heads,
        input_format=cfg.INPUT.FORMAT,
        vis_period=cfg.VIS_PERIOD,
        pixel_mean=cfg.MODEL.CLIP.PIXEL_MEAN,
        pixel_std=cfg.MODEL.CLIP.PIXEL_STD,
        clip_crop_region_type=cfg.MODEL.CLIP.CROP_REGION_TYPE,
        use_clip_attpool=cfg.MODEL.ROI_HEADS.NAME,
    )
