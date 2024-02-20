# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances

__all__ = ["FastRCNNOutputLayers"]

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # "gt_classes" exists if and only if training. But other gt fields may
            # not necessarily exist in training for images that have no groundtruth.
            if proposals[0].has("gt_classes"):
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = [
                    p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes for p in proposals
                ]
                self.gt_boxes = box_type.cat(gt_boxes)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  # no instances found

    def predict_boxes(self):
        """
        Deprecated
        """
        pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
        return pred.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            clip_cls_emb: tuple = (False, None),
            openset_test: None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            clip_cls_emb (Tuple): (use_clip_cls_emb (bool), pretrained_ckpt of concept_emb_model(str),
                                   RoIhead type(str), feature embedding dimension)
            openset_test (Tuple): (is_openset, pretrained_ckpt of concept_embed_path (dtr),
                                   temperature (float))
        """
        super().__init__()
        self.box2box_transform = box2box_transform
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        # RegionCLIP
        self.num_classes = num_classes
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        self.use_clip_cls_emb = clip_cls_emb[0]
        if self.use_clip_cls_emb:  # use CLIP text embeddings as classifier's weights
            input_size = clip_cls_emb[3]
            text_emb_require_grad = False
            self.use_bias = False
            self.temperature = openset_test[2]  # 0.01 is default for CLIP

            # class embedding
            self.cls_score = nn.Linear(input_size, num_classes, bias=self.use_bias)
            with torch.no_grad():
                if clip_cls_emb[1] is not None:  # it could be None during region feature extraction
                    # [num_classes, 1024] for RN50, [num_classes, 640] for RN50x4
                    pre_computed_w = torch.load(clip_cls_emb[1], map_location=self.cls_score.weight.device)
                    self.cls_score.weight.copy_(pre_computed_w)
                self.cls_score.weight.requires_grad = text_emb_require_grad  # freeze embeddings
                if self.use_bias:
                    nn.init.constant_(self.cls_score.bias, 0)

            # background embedding
            self.cls_bg_score = nn.Linear(input_size, 1, bias=self.use_bias)
            with torch.no_grad():
                nn.init.constant_(self.cls_bg_score.weight, 0)  # zero embeddings
                self.cls_bg_score.weight.requires_grad = text_emb_require_grad
                if self.use_bias:
                    nn.init.constant_(self.cls_bg_score.bias, 0)

            # class embedding during test
            self.test_cls_score = None
            if openset_test[1] is not None:  # open-set test enabled
                pre_computed_w = torch.load(openset_test[1])  # [#openset_test_num_cls, 1024] for RN50
                self.openset_test_num_cls = pre_computed_w.size(0)
                self.test_cls_score = nn.Linear(input_size, self.openset_test_num_cls, bias=self.use_bias)
                self.test_cls_score.weight.requires_grad = False  # freeze embeddings
                with torch.no_grad():
                    self.test_cls_score.weight.copy_(pre_computed_w)
                    if self.use_bias:
                        nn.init.constant_(self.test_cls_score.bias, 0)
        else:  # regular classification layer
            self.cls_score = nn.Linear(input_size, num_classes + 1)  # one background class (hence + 1)
            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)

        # box regression layer
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # RegionCLIP
            "clip_cls_emb": (cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER, cfg.MODEL.CLIP.TEXT_EMB_PATH,
                             cfg.MODEL.ROI_HEADS.NAME, cfg.MODEL.CLIP.TEXT_EMB_DIM),
            "openset_test": (cfg.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES, cfg.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH, \
                             cfg.MODEL.CLIP.CLSS_TEMP, cfg.MODEL.CLIP.FOCAL_SCALED_LOSS)
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # use clip text embeddings as classifier's weights
        if self.use_clip_cls_emb:
            normalized_x = F.normalize(x, p=2.0, dim=1)

            if self.test_cls_score is not None:
                # open-set inference enabled
                cls_scores = normalized_x @ F.normalize(self.test_cls_score.weight, p=2.0, dim=1).t()
                if self.use_bias:
                    cls_scores += self.test_cls_score.bias

            else:
                # training or closed-set model inference
                cls_scores = normalized_x @ F.normalize(self.cls_score.weight, p=2.0, dim=1).t()
                if self.use_bias:
                    cls_scores += self.cls_score.bias

            # background class (zero embeddings)
            bg_score = self.cls_bg_score(normalized_x)
            if self.use_bias:
                bg_score += self.cls_bg_score.bias

            scores = torch.cat((cls_scores, bg_score), dim=1)
            scores = scores / self.temperature

        else:
            # regular classifier
            scores = self.cls_score(x)

        # box regression
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

