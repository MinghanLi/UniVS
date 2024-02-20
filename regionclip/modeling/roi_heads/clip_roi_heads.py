# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec

from detectron2.modeling import (
    ROI_HEADS_REGISTRY,
    ROIHeads,
)

from detectron2.modeling.poolers import ROIPooler

from .fast_rcnn import FastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class CLIPRes5ROIHeads(ROIHeads):
    """
    Created for CLIP ResNet. This head uses the last resnet layer from backbone.
    Extended from Res5ROIHeads in roi_heads.py
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        res5: None,
        box_predictor: nn.Module,
        mask_head: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of backbone feature map names to use for
                feature extraction
            pooler (ROIPooler): pooler to extra region features from backbone
            res5 (nn.Sequential): a CNN to compute per-region features, to be used by
                ``box_predictor`` and ``mask_head``. Typically, this is a "res5"
                block from a ResNet.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        self.res5 = res5  #  None, this head uses the res5 from backbone
        self.box_predictor = box_predictor
        self.mask_on = mask_head is not None

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # cls._build_res5_block(cfg)
        ret["res5"], out_channels = None, cfg.MODEL.CLIP.RESNETS_RES2_OUT_CHANNELS * 8
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        return ret

    def _shared_roi_transform(self, features, boxes, backbone_res5):
        x = self.pooler(features, boxes)
        return backbone_res5(x)

    def forward(self, features, proposal_boxes, res5=None, attnpool=None):
        """
        See :meth:`ROIHeads.forward`.
        feature:
        proposal_boxes: List[boxes, boxes, ...]
        """

        # the length of list equals to batch size
        num_boxes_per_img = [boxes.tensor.shape[0] for boxes in proposal_boxes]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], [boxes for boxes in proposal_boxes], res5
        )

        if attnpool:
            # att pooling
            att_feats = attnpool(box_features)
            predictions = self.box_predictor(att_feats)
        else:
            # mean pooling
            predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        # predictions: List[scores, scores, ...]
        cls_scores, _ = predictions
        cls_log_probs = F.log_softmax(cls_scores, dim=-1)
        cls_log_probs = cls_log_probs.split(num_boxes_per_img, dim=0)

        return cls_log_probs

