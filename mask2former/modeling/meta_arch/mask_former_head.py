# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # NOTE Modified by Sukjun Hwang: Issues with recent detectron2 versions.
        version = 2  # local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        pixel_decoder_name: str,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        # frozen decoders
        frozen_pixel_decoder: bool=False,
        frozen_mask_convs: bool=False,
        frozen_predictor: bool=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.pixel_decoder_name = pixel_decoder_name
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

        self.frozen_pixel_decoder = frozen_pixel_decoder
        self.frozen_mask_convs = frozen_mask_convs
        self.frozen_predictor = frozen_predictor
        self._freeze_decoders()
    
    def _freeze_decoders(self):
        if self.frozen_pixel_decoder:
            for name, param in self.pixel_decoder.named_parameters():
                if self.frozen_mask_convs:
                    param.requires_grad = False
                elif 'input_proj' in name or 'transformer.encoder' in name:
                    param.requires_grad = False

        if self.frozen_predictor:
            self.predictor.eval()
            for param in self.predictor.parameters():
                param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "pixel_decoder_name": cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
            "frozen_pixel_decoder": cfg.MODEL.SEM_SEG_HEAD.FROZEN_PIXEL_DECODER,
            "frozen_mask_convs": cfg.MODEL.SEM_SEG_HEAD.FROZEN_MASK_CONVS,
            "frozen_predictor": cfg.MODEL.SEM_SEG_HEAD.FROZEN_PREDICTOR,
        }

    def forward(self, features, mask=None, targets=None):
        return self.layers(features, mask, targets)

    def layers(self, features, mask=None, targets=None):
        if self.pixel_decoder_name == "MSDeformAttnPixelDecoder":
            mask_features, mask_features_bfe_conv, transformer_encoder_features, multi_scale_features = \
                self.pixel_decoder.forward_features(features)

            if self.transformer_in_feature == "multi_scale_pixel_decoder":
                predictions = self.predictor(multi_scale_features, mask_features, mask_features_bfe_conv, mask, targets)
            else:
                if self.transformer_in_feature == "transformer_encoder":
                    assert (
                        transformer_encoder_features is not None
                    ), "Please use the TransformerEncoderPixelDecoder."
                    predictions = self.predictor(transformer_encoder_features, mask_features, mask)
                elif self.transformer_in_feature == "pixel_embedding":
                    predictions = self.predictor(mask_features, mask_features, mask)
                else:
                    predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
            return predictions
        
        elif self.pixel_decoder_name == "MSDeformAttnPixelDecoderVL":
            task = targets[0]["task"]
            if task == "detection":
                lang_features = torch.stack([t["clip_cls_text_emb"] for t in targets])  # B x num_cates x d_text
                num_frames = targets[0]["num_frames"]
                lang_features = lang_features[:, None].repeat(1,num_frames,1,1).flatten(0,1)
            elif task == "grounding":
                lang_features = torch.stack([
                    torch.stack([t["exp_sentence_feats"][:, None], t["exp_word_feats"]], dim=1) 
                    for t in targets
                ])  
                # B x num_exps x (1+77) x T x d_text -> BT x [num_exps * (1+77)] x d_text
                lang_features = lang_features.flatten(1,2).transpose(1,2).flatten(0,1)  
            else:
                lang_features = None

            # vision and language fusion in UNINEXT
            mask_features, mask_features_bfe_conv, transformer_encoder_features, multi_scale_features, lang_features = \
                self.pixel_decoder.forward_features(features, lang_features)

            assert self.transformer_in_feature == "multi_scale_pixel_decoder"  # mask2former 
            predictions = self.predictor(lang_features, multi_scale_features, mask_features, mask_features_bfe_conv, mask, targets)
            return predictions
        else:
             raise ValueError
