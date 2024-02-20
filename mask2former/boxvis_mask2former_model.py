import os.path
import time
from typing import Tuple
import math
import skimage
# support color space pytorch, https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html#rgb_to_lab
from kornia import color
from einops import rearrange
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

from scipy.optimize import linear_sum_assignment

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from mask2former.utils.box_ops import box_xyxy_to_cxcywh


@META_ARCH_REGISTRY.register()
class BoxVIS_MaskFormer(nn.Module):
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
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        test_topk_per_image: int,
        # vita
        num_frames: int,
        num_classes: int,
        is_multi_cls: bool,
        apply_cls_thres: float,
        freeze_detector: bool,
        test_run_chunk_size: int,
        test_interpolate_chunk_size: int,
        is_coco: bool,
        # boxvis
        boxvis_enabled: bool,
        boxvis_bottom_pixels_removed: int,
        boxvis_pairwise_size: int,
        boxvis_pairwise_dilation: int,
        boxvis_pairwise_color_thresh: float,
        boxvis_boxdet_on: bool,
        boxvis_ema_enabled: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.test_topk_per_image = test_topk_per_image

        # vita hyper-parameters
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres

        if freeze_detector:
            for name, p in self.named_parameters():
                if not "vita_module" in name:
                    p.requires_grad_(False)
        self.test_run_chunk_size = test_run_chunk_size
        self.test_interpolate_chunk_size = test_interpolate_chunk_size

        self.is_coco = is_coco

        # boxvis
        self.boxvis_enabled = boxvis_enabled
        self.boxvis_bottom_pixels_removed = boxvis_bottom_pixels_removed
        self.boxvis_pairwise_size = boxvis_pairwise_size
        self.boxvis_pairwise_dilation = boxvis_pairwise_dilation
        self.boxvis_pairwise_color_thresh = boxvis_pairwise_color_thresh

        self.boxvis_boxdet_on = boxvis_boxdet_on
        self.boxvis_ema_enabled = boxvis_ema_enabled

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        box_weight = cfg.MODEL.BoxVIS.BOX_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_box=box_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            boxvis_enabled=cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            boxvis_pairwise_size=cfg.MODEL.BoxVIS.PAIRWISE_SIZE,
            boxvis_pairwise_dilation=cfg.MODEL.BoxVIS.PAIRWISE_DILATION,
            boxvis_pairwise_color_thresh=cfg.MODEL.BoxVIS.PAIRWISE_COLOR_THRESH,
            boxvis_boxdet_on=cfg.MODEL.BoxVIS.BOXDET_ON,
        )

        boxvis_enabled = cfg.MODEL.BoxVIS.BoxVIS_ENABLED
        boxvis_boxdet_on = cfg.MODEL.BoxVIS.BOXDET_ON
        if boxvis_enabled:
            weight_dict = {"loss_ce": class_weight, "loss_mask_proj": dice_weight, "loss_mask_pair": 1.,
                           "loss_mask": 0.5 * mask_weight, "loss_dice": 0.5 * dice_weight}
            if boxvis_boxdet_on:
                weight_dict.update({"loss_box": box_weight, "loss_giou": box_weight})
        else:
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes", "masks"] if boxvis_enabled and boxvis_boxdet_on else ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            vita_last_layer_num=cfg.MODEL.VITA.LAST_LAYER_NUM,
            boxvis_enabled=cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            boxvis_pairwise_size=cfg.MODEL.BoxVIS.PAIRWISE_SIZE,
            boxvis_pairwise_dilation=cfg.MODEL.BoxVIS.PAIRWISE_DILATION,
            boxvis_pairwise_color_thresh=cfg.MODEL.BoxVIS.PAIRWISE_COLOR_THRESH,
            boxvis_pairwise_warmup_iters=cfg.MODEL.BoxVIS.PAIRWISE_WARMUP_ITERS,
            boxvis_boxdet_on=cfg.MODEL.BoxVIS.BOXDET_ON,
            boxvis_ema_enabled=cfg.MODEL.BoxVIS.EMA_ENABLED,
        )
        num_classes = sem_seg_head.num_classes

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # vita
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": num_classes,
            "is_multi_cls": cfg.MODEL.VITA.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.VITA.APPLY_CLS_THRES,
            "freeze_detector": cfg.MODEL.VITA.FREEZE_DETECTOR,
            "test_run_chunk_size": cfg.MODEL.VITA.TEST_RUN_CHUNK_SIZE,
            "test_interpolate_chunk_size": cfg.MODEL.VITA.TEST_INTERPOLATE_CHUNK_SIZE,
            "is_coco": cfg.DATASETS.TEST[0].startswith("coco"),
            # boxvis
            'boxvis_enabled': cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            'boxvis_bottom_pixels_removed': cfg.MODEL.BoxVIS.BOTTOM_PIXELS_REMOVED,
            'boxvis_pairwise_size': cfg.MODEL.BoxVIS.PAIRWISE_SIZE,
            'boxvis_pairwise_dilation': cfg.MODEL.BoxVIS.PAIRWISE_DILATION,
            'boxvis_pairwise_color_thresh': cfg.MODEL.BoxVIS.PAIRWISE_COLOR_THRESH,
            "boxvis_boxdet_on": cfg.MODEL.BoxVIS.BOXDET_ON,
            "boxvis_ema_enabled": cfg.MODEL.BoxVIS.EMA_ENABLED,
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
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            return self.train_model(batched_inputs)
        else:
            # NOTE consider only B=1 case.
            if self.is_coco:
                return self.inference(batched_inputs[0])
            else:
                return self.inference_video(batched_inputs[0])

    def train_model(self, batched_inputs):
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                if torch.rand(1) < 0.:
                    # convert RGB image to grayscale
                    images.append((color.rgb_to_grayscale(frame.to(self.device) / 255.) * 255.).repeat(3, 1, 1))
                else:
                    images.append(frame.to(self.device))
        images_norm = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_norm = ImageList.from_tensors(images_norm, self.size_divisibility)
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images_norm.tensor)
        outputs = self.sem_seg_head(features)[0]

        # mask classification target
        frame_targets, clip_targets = self.prepare_targets(batched_inputs, images)

        # bipartite matching-based loss
        losses, fg_indices = self.criterion(outputs, frame_targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

    def convert_box_to_mask(self, outputs_box: torch.Tensor, h: int, w: int):
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
        gt_x1 = grid_x >= outputs_box_wonorm[..., 0, None, None]
        lt_x2 = grid_x <= outputs_box_wonorm[..., 2, None, None]
        gt_y1 = grid_y >= outputs_box_wonorm[..., 1, None, None]
        lt_y2 = grid_y <= outputs_box_wonorm[..., 3, None, None]
        cropped_box_mask = gt_x1 & lt_x2 & gt_y1 & lt_y2

        return cropped_box_mask

    def prepare_targets(self, targets, images):
        # Note: images without MEAN and STD normalization
        BT, c, h_pad, w_pad = images.tensor.shape
        frame_gt_instances = []
        clip_gt_instances = []
        images = images.tensor.reshape(BT//self.num_frames, self.num_frames, -1, h_pad, w_pad)
        for targets_per_video, images_per_video in zip(targets, images):
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            gt_boxes_per_video = torch.zeros([_num_instance, self.num_frames, 4], dtype=torch.float32, device=self.device)

            gt_classes_per_video = targets_per_video["instances"][0].gt_classes.to(self.device)
            gt_ids_per_video, images_lab_color = [], []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)

                _update_box = box_xyxy_to_cxcywh(targets_per_frame.gt_boxes.tensor)[..., 2:].gt(0).all(dim=-1)
                box_normalizer = torch.as_tensor([w_pad, h_pad, w_pad, h_pad],
                                                 dtype=torch.float32, device=self.device).reshape(1, -1)
                gt_boxes_per_video[_update_box, f_i] = targets_per_frame.gt_boxes.tensor[_update_box] / box_normalizer  # xyxy

                if self.boxvis_enabled:
                    gt_masks_per_video[:, f_i] = self.convert_box_to_mask(gt_boxes_per_video[:, f_i], h_pad, w_pad)

                    # Note: check color channels should be rgb, which is controlled by cfg.INPUT.FORMAT
                    # Note: images_per_video without MEAN and STD normalization.
                    # https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html
                    images_lab = color.rgb_to_lab((images_per_video[f_i] / 255.).unsqueeze(0))  # 1x3xHxW
                    images_lab_color.append(images_lab)
                else:
                    if isinstance(targets_per_frame.gt_masks, BitMasks):
                        gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                    else:  # polygon
                        gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2, 3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long()  # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()          # N, num_frames
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()     # N, num_frames, H, W
            gt_boxes_per_video = gt_boxes_per_video[valid_bool_clip].float()     # N, num_frames, 4
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            display_mask = False
            if display_mask:
                import matplotlib.pyplot as plt
                from einops import rearrange
                i = int(torch.rand(1) * 100000)
                plt.imshow(images_per_video[0].byte().permute(1, 2, 0).cpu().numpy())
                plt.savefig('output/coco/boxvis_hard/' + str(i) + '_img.jpg')
                mask_show = gt_masks_per_video[:4 * (max(gt_masks_per_video.shape[0] // 4, 1)), 0]
                plt.imshow(
                    rearrange(mask_show, '(K R) H W -> (K H) (R W)', R=min(4, mask_show.shape[0])).cpu().numpy())
                plt.savefig('output/coco/boxvis_hard/' + str(i) + '_mask.jpg')

            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id

            clip_gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video, "masks": gt_masks_per_video,
                    "boxes": gt_boxes_per_video,
                    "video_len": targets_per_video["video_len"], "frame_idx": targets_per_video["frame_idx"],
                }
            )

            if self.boxvis_enabled:
                images_lab_color = torch.cat(images_lab_color, dim=0)  # num_frames, 3, h_pad, w_pad
                clip_gt_instances[-1]["image_lab_color"] = images_lab_color

            for f_i in range(self.num_frames):
                _cls = gt_classes_per_video.clone()
                _ids = gt_ids_per_video[:, f_i].clone()
                _mask = gt_masks_per_video[:, f_i].clone()
                _box = gt_boxes_per_video[:, f_i].clone()

                valid = _ids != -1
                frame_gt_instances.append({
                    "labels": _cls[valid],
                    "ids": _ids[valid],
                    "masks": _mask[valid],
                    "boxes": _box[valid]
                })

                if self.boxvis_enabled:
                    frame_gt_instances[-1]["image_lab_color"] = images_lab_color[f_i].clone()

        return frame_gt_instances, clip_gt_instances

    def inference(self, batched_inputs):
        to_store = self.device
        img_name = batched_inputs['file_names'][0].split('/')[-1][:-4]

        images = batched_inputs["image"]
        images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs, frame_queries, mask_features = self.sem_seg_head(features)

        mask_cls = outputs["pred_logits"][0]  # cQ, K+1
        mask_pred = outputs["pred_masks"][0]  # cQ, H, W

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1).flatten(0, 1)

        num_topk = self.test_topk_per_image
        scores_per_video, topk_indices = scores.flatten(0, 1).topk(num_topk, sorted=False)
        labels_per_video = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')
        mask_pred = mask_pred[topk_indices]

        # upsample masks
        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs.get("width", image_size[1])

        mask_pred = retry_if_cuda_oom(F.interpolate)(
            mask_pred.unsqueeze(1),
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        )  # cQ, 1, H, W

        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

        display_mask = False
        if display_mask:
            scores, idx = scores_per_video.sort(descending=True)
            root_dir = 'output/coco/boxvis_mask2former_r50_coco_f1/proj+pair/9_pairwise_200k/'
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            plt.imshow(batched_inputs["image"][0].byte().permute(1, 2, 0).cpu().numpy())
            plt.savefig(root_dir + img_name + '_img.jpg')
            plt.imshow(rearrange(mask_pred[idx[:12], 0].sigmoid().gt(0.5), '(K R) H W -> (K H) (R W)', K=3).cpu().numpy())
            plt.savefig(root_dir + img_name + '_mask.jpg')

        interim_mask_soft = mask_pred.sigmoid()
        interim_mask_hard = interim_mask_soft > 0.5
        numerator = (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
        denominator = interim_mask_hard.flatten(1).sum(1)
        scores_per_video *= (numerator / (denominator + 1e-6))

        mask_pred = F.interpolate(
            mask_pred, size=(out_height, out_width), mode="bilinear", align_corners=False
        ) > 0.
        mask_pred = mask_pred.to(to_store)

        valid = scores_per_video > 0.05
        scores_per_video = scores_per_video[valid]
        labels_per_video = labels_per_video[valid]
        mask_pred = mask_pred[valid]

        result = Instances((out_height, out_width))

        result.pred_masks = mask_pred[:, 0].float()  # T=1 for COCO
        # result.pred_boxes = Boxes(mask_box)
        # Uncomment the following to get boxes from masks (this is slow)
        result.pred_boxes = BitMasks(mask_pred[:, 0] > 0).get_bounding_boxes()

        result.scores = scores_per_video
        result.pred_classes = labels_per_video

        processed_results = [{"instances": result}]

        return processed_results

    def inference_video(self, batched_inputs):
        num_frames = len(batched_inputs["image"])
        to_store = self.device

        # convert RGB image to grayscale
        # batched_inputs["image"] = [(color.rgb_to_grayscale(x / 255.) * 255.).repeat(3, 1, 1)
        #                            for x in batched_inputs["image"]]

        out_logits = []
        out_masks = []
        out_embds = []
        for i in range(math.ceil(num_frames / self.test_run_chunk_size)):
            images = batched_inputs["image"][i * self.test_run_chunk_size: (i + 1) * self.test_run_chunk_size]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)
            outputs, frame_queries, mask_features = self.sem_seg_head(features)

            pred_logits, pred_masks = outputs['pred_logits'], outputs['pred_masks']

            # pred_logits: t q k
            # pred_masks: t q h w
            # pred_embeds: t q c
            pred_logits = list(torch.unbind(pred_logits))
            pred_masks = list(torch.unbind(pred_masks))
            frame_queries = list(torch.unbind(frame_queries))

            if len(out_embds) == 0:
                out_logits.append(pred_logits[0])
                out_masks.append(pred_masks[0])
                out_embds.append(frame_queries[0])

            for i in range(1, len(pred_logits)):
                indices = self.match_from_embds(out_embds[-1], frame_queries[i])

                out_logits.append(pred_logits[i][indices, :])
                out_masks.append(pred_masks[i][indices, :, :])
                out_embds.append(frame_queries[i][indices, :])

        mask_cls = torch.stack(out_logits, dim=1).mean(dim=1)  # q t k -> q k
        mask_pred = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        # upsample masks
        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation
        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs.get("width", image_size[1])

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1).flatten(0, 1)

        num_topk = min(self.test_topk_per_image, max(scores.gt(0.05).sum(), 1))
        scores_per_video, topk_indices = scores.flatten(0, 1).topk(num_topk, sorted=False)
        labels_per_video = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')
        mask_pred = mask_pred[topk_indices]
        mask_pred = retry_if_cuda_oom(F.interpolate)(
            mask_pred,
            size=interim_size,
            mode="bilinear",
            align_corners=False,
        )  # cQ, t, H, W

        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

        display_mask = False
        if display_mask:
            self.display_masks(batched_inputs, scores_per_video, mask_pred)

        interim_mask_soft = mask_pred.sigmoid()
        interim_mask_hard = interim_mask_soft > 0.5
        numerator = (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
        denominator = interim_mask_hard.flatten(1).sum(1)
        scores_per_video = scores_per_video * (numerator / (denominator + 1e-6))

        mask_pred = F.interpolate(
            mask_pred, size=(out_height, out_width), mode="bilinear", align_corners=False
        ) > 0.
        mask_pred = mask_pred.to(to_store)

        processed_results = {
            "image_size": (out_height, out_width),
            "pred_scores": scores_per_video.tolist(),
            "pred_labels": labels_per_video.tolist(),
            "pred_masks": mask_pred.cpu(),
        }

        return processed_results

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

    def display_masks(self, batched_inputs, scores_per_video, masks_pred):
        video_name = batched_inputs['file_names'][0].split('/')[-2]
        root_dir = 'output/coco/minvis_r50_ytvis21_f1_nolsj/9/' + video_name + '/'
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        scores, idx = scores_per_video.sort(descending=True)
        masks_pred = masks_pred[idx]

        color = torch.tensor([0.850, 0.325, 0.098]).reshape(-1, 1, 1)
        r = min(6, masks_pred.shape[1])
        num_img = r*(masks_pred.shape[1]//r)
        for i in range(len(scores)):
            plt.imshow(rearrange(masks_pred[i, :num_img].sigmoid().gt(0.5),
                                 '(K R) H W -> (K H) (R W)', R=r).cpu().numpy())
            plt.savefig(root_dir + str(i) + '_mask.jpg')

            for t in range(num_img):
                img = batched_inputs["image"][t].float() / 255.
                img = F.interpolate(img.unsqueeze(0), masks_pred.shape[-2:],
                                    mode='bilinear', align_corners=False).squeeze(0)
                m = masks_pred[i, t].unsqueeze(0).gt(0).cpu()
                img = (img * 0.4 + color * 0.6).clamp(min=0, max=1) * m + img * (~m)
                plt.imshow(img.permute(1, 2, 0).numpy())
                plt.savefig(root_dir + str(i) + '_' + str(t) + '_img.jpg')

