import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch
import torch.nn.functional as F

import os
from PIL import Image
import glob

import pycocotools.mask as mask_util
from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

# from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog

from fvcore.transforms.transform import HFlipTransform
from .augmentation import build_augmentation

from univs.data import detection_utils as utils
from univs.modeling.language import clean_string_exp

__all__ = ["UniVidDatasetMapper"]

def clean_strings(strings):
    unexpected = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "(", ")"]
    if isinstance(strings, list):
        for i, string in enumerate(strings):
            string = ' '.join(string.split('_'))
            string_l = list(string)
            string_l = [word for word in string_l if word not in unexpected]
            strings[i] = clean_string_exp(''.join(string_l))

        return strings

    else:
        strings = ' '.join(strings.split('_'))
        string_l = list(strings)
        string_l = [word for word in string_l if word not in unexpected]
        string_l = clean_string_exp(''.join(string_l))

        return string_l

def filter_empty_instances_soft(instances, by_box=True, by_mask=True, box_threshold=1e-5):
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

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1 # invalid instances are marked with -1
    return instances


def _get_dummy_anno(num_classes=-1, has_mask=True, has_expression=False):
    anno = {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
    }
    if has_mask:
        anno["segmentation"] = [np.array([0.0] * 6)]
    if has_expression:
        anno["expressions"] = []
        anno["exp_id"] = -1
    return anno


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


"""Unified DatasetMapper for video-level tasks"""
class UniVidDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
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
        sampling_frame_range_mot: int = 3,
        sampling_frame_range_sot: int = 200,
        sampling_interval: int = 1,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        dataset_name: str = "",
        test_categories=None,
        multidataset=False,
        prompt_type: str = "",
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
        self.multidataset           = multidataset
        self.augmentations = T.AugmentationList(augmentations)

        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = max(sampling_frame_num, sampling_frame_video_num)
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_range_mot = sampling_frame_range_mot
        self.sampling_frame_range_sot = sampling_frame_range_sot
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        self.dataset_name           = dataset_name
        self.prompt_type            = prompt_type

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, dataset_name: str = "", test_categories=None):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_video_num = cfg.INPUT.SAMPLING_FRAME_VIDEO_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_frame_range_mot = cfg.INPUT.SAMPLING_FRAME_RANGE_MOT
        sampling_frame_range_sot = cfg.INPUT.SAMPLING_FRAME_RANGE_SOT
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_video_num": sampling_frame_video_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_range_mot": sampling_frame_range_mot,
            "sampling_frame_range_sot": sampling_frame_range_sot,
            "sampling_interval": sampling_interval,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "dataset_name": dataset_name,
            "test_categories": test_categories,
            "multidataset": cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler",
            "prompt_type": cfg.MODEL.UniVS.PROMPT_TYPE,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO: consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        dataset_name = dataset_dict["dataset_name"]

        video_length = dataset_dict["length"]
        is_image_data = video_length == 1
        if self.is_train:
            if video_length == 1:
                pseudo_video_length = self.sampling_frame_num * self.sampling_interval
                # get pseudo video from static image
                dataset_dict["file_names"] = dataset_dict["file_names"] * pseudo_video_length
                dataset_dict["annotations"] = dataset_dict["annotations"] * pseudo_video_length
                video_length = pseudo_video_length
                if "pan_seg_file_names" in dataset_dict:
                    dataset_dict["pan_seg_file_names"] = dataset_dict["pan_seg_file_names"] * pseudo_video_length

            ref_frame = random.randrange(video_length)
            if dataset_name.startswith("mots"):
                sampling_frame_range = self.sampling_frame_range_mot
            elif dataset_name.startswith("sot") or dataset_name.startswith("burst"):
                sampling_frame_range = self.sampling_frame_range_sot
            else:
                sampling_frame_range = self.sampling_frame_range
            start_idx = max(0, ref_frame-sampling_frame_range)
            start_interval = max(0, ref_frame-self.sampling_interval+1)
            end_idx = min(video_length, ref_frame+sampling_frame_range + 1)
            end_interval = min(video_length, ref_frame+self.sampling_interval)

            candidate_frame_idxs = list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))
            if len(candidate_frame_idxs) == 0:
                selected_idx = [ref_frame] * (self.sampling_frame_num - 1)
            else:
                selected_idx = np.random.choice(
                    np.array(candidate_frame_idxs),
                    self.sampling_frame_num - 1,
                ).tolist()
            selected_idx = selected_idx + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        # selected_idx is a List of length self.sampling_frame_num
        video_annos = dataset_dict.pop("annotations", None)  # List
        file_names = dataset_dict.pop("file_names", None)  # List
        pan_seg_file_names = dataset_dict.pop("pan_seg_file_names", None)  # List

        _ids = set()
        for frame_idx in selected_idx:
            for anno in video_annos[frame_idx]:
                if "expressions" in anno:
                    _ids.update({anno["exp_id"]})
                elif anno['bbox'] or anno['segmentation']:
                    _ids.update({anno["id"]})
        ids = dict()
        for i, _id in enumerate(_ids):
            ids[_id] = i  # original instance id -> zero-based

        if self.is_train and (len(ids) == 0):
            print('No objects in', dataset_name, 'to re-load...')
            return None

        dataset_dict["video_len"] = video_length
        dataset_dict["frame_indices"] = list(selected_idx)
        dataset_dict["image"] = []
        dataset_dict["image_padding_mask"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []

        task = dataset_dict["task"]
        selected_augmentations = self.augmentations

        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])
            
            # Read image
            try:
                image = utils.read_image(file_names[frame_idx], format=self.image_format)
                # the image in entityseg dataset may has low-resolution
                if dataset_name.startswith('entityseg'):
                    if (dataset_dict["height"], dataset_dict["width"]) != image.shape:
                        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                        image = F.interpolate(
                            image, size=(dataset_dict["height"], dataset_dict["width"]), 
                            mode='bilinear', align_corners=False
                        )
                        image = image.squeeze(0).permute(1, 2, 0).numpy()  # Convert back to NumPy array
            except:
                if 'entityseg' not in file_names[frame_idx]:
                    print("Not find image:", file_names[frame_idx], "reload...")
                # there are some images/videos have not been downloaded..
                return None
            original_image_wh = (dataset_dict["width"], dataset_dict["height"])
            if self.is_train:
                try:
                    utils.check_image_size(dataset_dict, image)
                except:
                    print("Mask shape:", (dataset_dict["height"], dataset_dict["width"]), "Image shape:", image.shape)
                    print(f"There are some videos with inconsistent resolutions {file_names[0]}, reload...")
                    # there are some videos with inconsistent resolutions...
                    # eg. GOT10K/val/GOT-10k_Val_000137
                    return None

            image_padding_mask = np.ones_like(image)

            aug_input = T.AugInput(image)
            transforms = selected_augmentations(aug_input)

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

            # for evaluation
            if not self.is_train:
                if task in ["detection"]:
                    continue
                
                elif task  == "grounding" and frame_idx == selected_idx[0]:
                    expressions = dataset_dict["expressions"]
                    if isinstance(expressions[0], list):
                        expressions = sum(expressions, [])
                    dataset_dict["expressions"] = transform_expressions(expressions, transforms, self.is_train)
                    dataset_dict["exp_obj_ids"] = dataset_dict["exp_id"]

                    if 'davis' in dataset_name:
                        # save predicted masks with palette 
                        dataset_dict["mask_palette"] = self.get_palette(file_names, dataset_name)

                    continue

                elif task == "sot":
                    # for SOT and VOS, we need the box anno in the 1st frame during inference
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

                    instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                    if instances.has("gt_masks"):
                        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

                    # add ori_id for VOS inference
                    ori_id_list = [int(x["ori_id"]) if "ori_id" in x else int(x["id"]) for x in annos]
                    instances.ori_ids = ori_id_list
                    dataset_dict["instances"].append(instances)

                    # get palette for evaluation
                    if len(dataset_dict["file_names"]) == 1:
                        dataset_dict["mask_palette"] = self.get_palette(file_names, dataset_name)

                    continue
            
            has_mask = dataset_dict["has_mask"]
            has_caption = dataset_dict["has_caption"]
            if has_caption:
                caption = dataset_dict["caption"]
                tokens_positive_eval = dataset_dict["tokens_positive_eval"]
                dataset_dict["phrases"] = [
                    caption[tokens_idx[0][0]:tokens_idx[0][1]] for tokens_idx in tokens_positive_eval
                ]

            needs_annos = self.prompt_type in {'points', 'boxes', 'masks'}
            if (video_annos is None) or (not self.is_train and not needs_annos):
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

            # load gt panoptic segmentation from .PNG
            if pan_seg_file_names:
                try:
                    pan_seg_gt = utils.read_image(pan_seg_file_names[frame_idx], format="RGB")
                except:
                    print("Not find gt panoptic masks:", pan_seg_file_names[frame_idx], "reload...")
                    # there are some images/videos have not been downloaded..
                    return None

                # save thing or stuff category ids
                dataset_dict["metadata"] = dataset_dict["metadata"]

                # apply the same transformation to panoptic segmentation
                pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
                from panopticapi.utils import rgb2id
                pan_seg_gt = rgb2id(pan_seg_gt).astype("long")

                # check_panoptic_seg(image, pan_seg_gt, [_anno["id"] for _anno in annos], file_names[frame_idx])
                # prepare per-obj binary masks
                for _anno in annos:
                    if _anno.get("iscrowd", 0) == 0:
                        _mask = np.asfortranarray(pan_seg_gt == _anno["id"])
                        _anno["segmentation"] = mask_util.encode(_mask)  # binary mask to RLE
            
            sorted_annos = [
                _get_dummy_anno(has_mask=has_mask, has_expression=task == "grounding")
                for _ in range(len(ids))
            ]

            for _anno in annos:
                _id = _anno["exp_id"] if task == "grounding" else _anno["id"]
                if _id in ids:
                    idx = ids[_id]  # original id -> zero-based id
                    if _anno['bbox'] is not None or _anno['segmentation'] is not None:
                        sorted_annos[idx] = _anno

            _gt_ids = [
                int(_anno["exp_id"]) if task == "grounding" else int(_anno["id"])
                for _anno in sorted_annos
            ]
            if task == "grounding":
                # in Ref-COCO or Ref-ytbvos, per object ("id") has multiple expressions ("exp_id")
                _gt_obj_ids = [int(_anno["id"]) for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")

            if len(instances):
                num_objs = instances.gt_masks.tensor.shape[0] if has_mask else instances.gt_boxes.tensor.shape[0]
                instances.positive_map = torch.ones((num_objs, 1), dtype=torch.bool)
            else:
                print("invalid instance is found:", instances, annos, dataset_name,
                      file_names[frame_idx])

            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = filter_empty_instances_soft(copy.deepcopy(instances))
                    
            dataset_dict["instances"].append(instances)

            if len(sorted_annos) and frame_idx == selected_idx[0]:
                if task == "grounding":
                    assert "expressions" in sorted_annos[0]
                    # List: [obj1_exp, obj2_exp, ....]
                    expressions_list = [annos["expressions"] for annos in sorted_annos]
                    dataset_dict["expressions"] = transform_expressions(expressions_list, transforms, self.is_train)
                    dataset_dict["exp_obj_ids"] = _gt_obj_ids
                elif has_caption:
                     # convert phase tokens to ids
                    dataset_dict['token_ids'] = [
                        [dataset_dict["tokens_positive_eval"].index([tokens] if isinstance(tokens[0], int) else tokens) 
                        for tokens in _anno["tokens_positive"]]
                        for _anno in sorted_annos
                    ]

        if self.is_train and task == "grounding":
            if len(dataset_dict["expressions"]) == 0:
                return None

        if self.is_train and has_mask and len(dataset_dict["instances"]):
            # there are so many objects in SA1B, remove empty objects to speed up training and save memory
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

            dataset_dict["instances"] = [
                targets_per_frame[valid_idxs] for targets_per_frame in dataset_dict["instances"]
            ]

            if task == "grounding" and len(dataset_dict["expressions"]):
                dataset_dict["expressions"] = [dataset_dict["expressions"][_i] for _i in valid_idxs.tolist()]
                dataset_dict["exp_obj_ids"] = [dataset_dict["exp_obj_ids"][_i] for _i in valid_idxs.tolist()]
        
        return dataset_dict

    def get_palette(self, file_names, dataset_name=None):
        anno_dir = file_names[0].split('/')[:-1]
        
        if dataset_name is not None and "davis" in dataset_name:
            anno_dir[anno_dir.index('JPEGImages')] = 'Annotations'
        else:
            anno_dir[-2] = 'Annotations'
        anno_dir = '/'.join(anno_dir)

        file_names = glob.glob('/'.join([anno_dir, "*.png"]))
        file_names = sorted(file_names, key=lambda f: int(f.split("/")[-1][:3]))
        if len(file_names) == 0:
            return None
        
        im = Image.open(file_names[0]).convert('P')
        palette = im.getpalette()

        return palette

def transform_expressions(expressions_list, transforms, is_train):
        #print(expressions_list, [isinstance(x, HFlipTransform) for x in transforms])
        expression_objs = []
        for expressions in expressions_list:
            if len(expressions) == 0:
                expression_objs.append("")
                continue

            # pick one expression if there are multiple expressions
            if is_train:
                expression = expressions[np.random.choice(len(expressions))]
                expression = clean_strings(expression)
            else:
                if isinstance(expressions[0], list):
                    # for refdavis, the json has been preprocessed
                    # so "expressions": [["exp1", "exp2", ...]]
                    expression = clean_strings(expressions[0])  # ["exp1", "exp2", ...]
                else:
                    # for refcoco and refytvos, the json has been preprocessed
                    # so only one "expressions": ["exp1"]
                    expression = clean_strings(expressions)  # ["exp1"]

            # deal with hflip for expression
            hflip_flag = False
            for x in transforms:
                if isinstance(x, HFlipTransform):
                    hflip_flag = True
                    break
            if hflip_flag:
                if isinstance(expression, list):
                    expression = [e.replace('left', '@').replace('right', 'left').replace('@', 'right') for e in expression]
                else:
                    expression = expression.replace('left', '@').replace('right', 'left').replace('@', 'right')

            expression_objs.append(expression)
        
        return expression_objs


def check_panoptic_seg(image, pan_seg_gt, ids_list, file_name):
    import matplotlib.pyplot as plt
    out_dir = os.path.join("output/visual/pan_seg_gt/", file_name)
    os.makedirs(out_dir, exist_ok=True)

    print(os.path.join(out_dir, 'image.jpg'))
    plt.imshow(image)
    plt.savefig(os.path.join(out_dir, 'image.jpg'))
    plt.clf()
    for ids in ids_list:
        mask = pan_seg_gt == ids
        plt.imshow(mask)
        plt.savefig(os.path.join(out_dir, str(ids)+'.jpg'))
        plt.clf()
    exit()


