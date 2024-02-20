import copy
import logging
import os
import random
import numpy as np
from typing import List, Union
import torch

try:
    import orjson as json
except:
    import json

from einops import rearrange
import matplotlib.pyplot as plt

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog

from pycocotools import mask as coco_mask

from univs.data import detection_utils as utils
from fvcore.transforms.transform import HFlipTransform
from .augmentation import build_augmentation, build_pseudo_augmentation

import re

__all__ = ["YTVISDatasetMapper", "CocoClipDatasetMapper", ]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())
    if instances.has("gt_classes"):
        r.append(instances.gt_classes != -1)

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno():
    return {
        "iscrowd": 0,
        "category_id": -1,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class YTVISDatasetMapper:
    """
    Similar to YTVISDatasetMapper, only add text expressions
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_video_num: int = -1, 
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        dataset_name: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num if sampling_frame_video_num == -1 else sampling_frame_video_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        self.dataset_name           = dataset_name

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, dataset_name=None):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_video_num = cfg.INPUT.SAMPLING_FRAME_VIDEO_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_video_num": sampling_frame_video_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "dataset_name": dataset_name,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            if video_length == 1:
                selected_idx = [ref_frame] * self.sampling_frame_num
            else:
                start_idx = max(0, ref_frame-self.sampling_frame_range)
                end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
                selected_idx = sorted(selected_idx)

            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict["annotations"]
        file_names = dataset_dict["file_names"]

        if video_annos is not None:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["video_len"] = len(video_annos)
        dataset_dict["frame_idx"] = list(selected_idx)
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])
            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore, it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            # if (video_annos is None) or (not self.is_train):
            if video_annos is None:
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno() for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                if _anno['bbox'] is not None or _anno['segmentation'] is not None:
                    sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            instances = filter_empty_instances(instances)

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)

            if not instances.has("gt_masks"):
                if instances.has("gt_boxes"):
                    n_inst = instances.gt_boxes.tensor.shape[0]
                    instances.gt_masks = BitMasks(torch.empty((n_inst, *image_shape)))
                else:
                    instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        # remove empty objects from Instance
        gt_ids_per_video = []
        for f_i, targets_per_frame in enumerate(dataset_dict["instances"]):
            gt_ids_per_video.append(targets_per_frame.gt_ids)
        gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
        valid_idxs = torch.nonzero((gt_ids_per_video >= 0).any(1)).reshape(-1)

        # to speed up training and save memory, there are so many objects in SA1B
        dataset_dict["instances"] = [
            targets_per_frame[valid_idxs] for targets_per_frame in dataset_dict["instances"]
        ]

        return dataset_dict


def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')


class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        sampling_frame_num: int = 2,
        sampling_frame_video_num: int = -1,
        sampling_frame_range: int = 5,
        dataset_name: str = 'coco_2017_train',
        num_pos_queries: int = 20,
        eval_load_annotations: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.sampling_frame_num     = sampling_frame_num if sampling_frame_video_num == -1 else sampling_frame_video_num
        self.sampling_frame_range   = sampling_frame_range
        self.dataset_name = dataset_name
        self.num_pos_queries = num_pos_queries

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, dataset_name: str = ""):
        if cfg.INPUT.SAMPLING_FRAME_NUM == 1:
            augs = build_augmentation(cfg, is_train)
        else:
            augs = build_pseudo_augmentation(cfg, is_train)
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_video_num = cfg.INPUT.SAMPLING_FRAME_VIDEO_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_video_num": sampling_frame_video_num,
            "sampling_frame_range": sampling_frame_range,
            "dataset_name": dataset_name,
            "num_pos_queries": cfg.MODEL.UniVS.NUM_POS_QUERIES,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
            or dataset_dict (str): annotation file names
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        is_sa1b = False
        if isinstance(dataset_dict, str):
            is_sa1b = True
            # for SA-1B, where dataset_dict is the name of annotation file
            image_root = 'datasets/sa_1b/images'
            anno_root = 'datasets/sa_1b/annotations'

            file = open('/'.join([anno_root, dataset_dict]), 'r').read()
            annotations = json.loads(file)

            # {
            #     "image": image_info,
            #     "annotations": [annotation],
            # }
            #
            # image_info
            # {
            #     "image_id": int,  # Image id
            #     "width": int,  # Image width
            #     "height": int,  # Image height
            #     "file_name": str,  # Image filename
            # }
            #
            # annotation
            # {
            #     "id": int,  # Annotation id
            #     "segmentation": dict,  # Mask saved in COCO RLE format.
            #     "bbox": [x, y, w, h],  # The box around the mask, in XYWH format
            #     "area": int,  # The area in pixels of the mask
            #     "predicted_iou": float,  # The model's own prediction of the mask's quality
            #     "stability_score": float,  # A measure of the mask's quality
            #     "crop_box": [x, y, w, h],  # The crop of the image used to generate the mask, in XYWH format
            #     "point_coords": [[x, y]],  # The point coordinates input to the model to generate the mask
            # }

            dataset_dict = annotations["image"]
            dataset_dict["file_name"] = os.path.join(image_root, dataset_dict["file_name"])
            dataset_dict["annotations"] = annotations["annotations"]
            for anno_dict in dataset_dict["annotations"]:
                anno_dict["bbox_mode"] = BoxMode.XYWH_ABS

            dataset_dict["dataset_name"] = "sa_1b"
            dataset_dict["task"] = "sot"
            dataset_dict["has_stuff"] = True

        else:
            dataset_dict["has_stuff"] = False
            if "dataset_name" not in dataset_dict:
                if "pan_seg_file_name" in dataset_dict:
                    dataset_dict["dataset_name"] = "coco_panoptic"
                    dataset_dict["has_stuff"] = True
                else:
                    dataset_dict["dataset_name"] = "coco"
            dataset_dict["task"] = "detection"
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        if is_sa1b and img_annos is not None and len(img_annos) > 100:
            # remove masks with low stability_score or predicted_iou
            img_annos= [anno for anno in img_annos if anno["stability_score"] > 0.97 and anno["predicted_iou"] > 0.9]

        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        if self.is_train:
            video_length = random.randrange(16, 49)
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
        else:
            video_length = self.sampling_frame_num
            selected_idx = list(range(self.sampling_frame_num))

        dataset_dict["has_mask"] = True
        dataset_dict["video_len"] = video_length
        dataset_dict["frame_indices"] = selected_idx
        dataset_dict["image"] = []
        dataset_dict["image_padding_mask"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        if not self.is_train and "panoptic" in dataset_dict["dataset_name"]:
            # panoptic evaluator needs file_name
            dataset_dict["file_name"] = file_name

        for i in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)
            image_padding_mask = np.ones_like(original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            image_shape = image.shape[:2]  # h, w

            image_padding_mask = transforms.apply_segmentation(image_padding_mask)

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore, it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
            dataset_dict["image_padding_mask"].append(torch.as_tensor(
                np.ascontiguousarray(1 - image_padding_mask[:, :, 0])
            ))

            if (img_annos is None) or (not self.is_train):
                continue
                
            _img_annos = []
            for obj_i, anno in enumerate(img_annos):
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]

            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            mask_format = "bitmask" if is_sa1b else "polygon"
            instances = utils.annotations_to_instances(annos, image_shape, mask_format)
            instances.gt_ids = torch.tensor(_gt_ids)
            if len(annos) == 0:
                dataset_dict["instances"].append(instances)
                continue

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if hasattr(instances, 'gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # NOTE we don't need boxes
            instances = filter_empty_instances(instances)

            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                if not is_sa1b:
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                    instances.gt_masks = gt_masks

            # no classes in sa1b data
            if not instances.has("gt_classes"):
                instances.gt_classes = torch.ones_like(instances.gt_ids) * -1

            dataset_dict["instances"].append(instances)

        if self.is_train:
            # remove empty objects from Instance
            gt_ids_per_video, gt_masks_per_video = [], []
            for f_i, targets_per_frame in enumerate(dataset_dict["instances"]):
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks = targets_per_frame.gt_masks.tensor
                else:  # polygon
                    gt_masks = targets_per_frame.gt_masks
                gt_masks_per_video.append(gt_masks.sum((-1,-2)) > 0)
            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            gt_masks_per_video = torch.stack(gt_masks_per_video, dim=1)
            valid_idxs = torch.nonzero((gt_ids_per_video >= 0).any(1) & gt_masks_per_video.any(1)).reshape(-1)
            # to speed up training and save memory, there are so many objects in SA1B
            dataset_dict["instances"] = [
                targets_per_frame[valid_idxs] for targets_per_frame in dataset_dict["instances"]
            ]

        # display_pseudo_clip_from_coco(dataset_dict["image"], file_name)
        return dataset_dict


def display_pseudo_clip_from_coco(images_list, file_name, output_path='output/pseudo_clip_from_coco/'):
    imgs = torch.stack(images_list)  # T, 3, H, W
    plt.imshow(rearrange(imgs.cpu().numpy(), 'T C H W -> H (T W) C'))
    plt.axis('off')

    os.makedirs(output_path, exist_ok=True)
    plt.savefig(output_path + file_name.split('/')[-1])
    plt.clf()

