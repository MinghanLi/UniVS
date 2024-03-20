import contextlib
import io
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

from .burst import _map_burst_to_lvis_v1_dict

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_ytvis_instances"]


YTVIS_CATEGORIES_2019 = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "giant_panda"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "lizard"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "parrot"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "skateboard"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "sedan"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "ape"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "snake"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "monkey"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "hand"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "rabbit"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "duck"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "cow"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "fish"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "train"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "horse"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "turtle"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "bear"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "motorbike"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "leopard"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "fox"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "deer"},
    {"color": [188, 208, 182], "isthing": 1, "id": 26, "name": "owl"},
    {"color": [145, 148, 174], "isthing": 1, "id": 27, "name": "surfboard"},
    {"color": [106, 0, 228], "isthing": 1, "id": 28, "name": "airplane"},
    {"color": [0, 0, 70], "isthing": 1, "id": 29, "name": "truck"},
    {"color": [199, 100, 0], "isthing": 1, "id": 30, "name": "zebra"},
    {"color": [166, 196, 102], "isthing": 1, "id": 31, "name": "tiger"},
    {"color": [110, 76, 0], "isthing": 1, "id": 32, "name": "elephant"},
    {"color": [133, 129, 255], "isthing": 1, "id": 33, "name": "snowboard"},
    {"color": [0, 0, 192], "isthing": 1, "id": 34, "name": "boat"},
    {"color": [183, 130, 88], "isthing": 1, "id": 35, "name": "shark"},
    {"color": [130, 114, 135], "isthing": 1, "id": 36, "name": "mouse"},
    {"color": [107, 142, 35], "isthing": 1, "id": 37, "name": "frog"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "eagle"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "earless_seal"},
    {"color": [255, 208, 186], "isthing": 1, "id": 40, "name": "tennis_racket"},
]


YTVIS_CATEGORIES_2021 = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "bear"},
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "car"},
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "cow"},
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "deer"},
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "dog"},
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "duck"},
    {"color": [174, 255, 243], "isthing": 1, "id": 11, "name": "earless_seal"},
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "elephant"},
    {"color": [73, 77, 174], "isthing": 1, "id": 13, "name": "fish"},
    {"color": [250, 170, 30], "isthing": 1, "id": 14, "name": "flying_disc"},
    {"color": [0, 125, 92], "isthing": 1, "id": 15, "name": "fox"},
    {"color": [107, 142, 35], "isthing": 1, "id": 16, "name": "frog"},
    {"color": [0, 82, 0], "isthing": 1, "id": 17, "name": "giant_panda"},
    {"color": [72, 0, 118], "isthing": 1, "id": 18, "name": "giraffe"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [255, 179, 240], "isthing": 1, "id": 20, "name": "leopard"},
    {"color": [119, 11, 32], "isthing": 1, "id": 21, "name": "lizard"},
    {"color": [0, 60, 100], "isthing": 1, "id": 22, "name": "monkey"},
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "motorbike"},
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "mouse"},
    {"color": [165, 42, 42], "isthing": 1, "id": 25, "name": "parrot"},
    {"color": [220, 20, 60], "isthing": 1, "id": 26, "name": "person"},
    {"color": [100, 170, 30], "isthing": 1, "id": 27, "name": "rabbit"},
    {"color": [183, 130, 88], "isthing": 1, "id": 28, "name": "shark"},
    {"color": [134, 134, 103], "isthing": 1, "id": 29, "name": "skateboard"},
    {"color": [5, 121, 0], "isthing": 1, "id": 30, "name": "snake"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "snowboard"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "squirrel"},
    {"color": [145, 148, 174], "isthing": 1, "id": 33, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 34, "name": "tennis_racket"},
    {"color": [166, 196, 102], "isthing": 1, "id": 35, "name": "tiger"},
    {"color": [0, 80, 100], "isthing": 1, "id": 36, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 37, "name": "truck"},
    {"color": [0, 143, 149], "isthing": 1, "id": 38, "name": "turtle"},
    {"color": [0, 228, 0], "isthing": 1, "id": 39, "name": "whale"},
    {"color": [199, 100, 0], "isthing": 1, "id": 40, "name": "zebra"},
]


def _get_ytvis_2019_instances_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_ytvis_2021_instances_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None,
                    has_mask=True, has_expression=False, has_caption=False,sot=False, 
                    pan_gt_root=None, has_stuff=False):
    from .ytvis_api.ytvos import YTVOS
    has_pan_mask = True if pan_gt_root is not None else False

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        metadata = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        if 'val' not in dataset_name and 'dev' not in dataset_name and 'flickr' not in dataset_name:
            # thing + stuff classes for pan. seg.
            metadata.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        # entityseg in [0, #categories-1], coco_panoptic not in [1, #categories]
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "burst" in dataset_name:
                logger.warning(
                    f"""
                    Category ids in {dataset_name} annotations are not in [1, #categories]! We'll apply the mapping from burst to lvis v1.
                    """
                )
                id_map = _map_burst_to_lvis_v1_dict
            elif "coco" not in dataset_name or "coco_panoptic" in dataset_name:
                logger.warning(
                    f"""
                    Category ids in {dataset_name} annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
                metadata.thing_dataset_id_to_contiguous_id = {v: i for i, v in enumerate(cat_ids)}
                id_map = {v: i + 1 for i, v in enumerate(cat_ids)}  # start from 1

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        if isinstance(vid_dict["file_names"], str):
            vid_dict["file_names"] = [vid_dict["file_names"]]

        record = {
            "file_names": [
                os.path.join(image_root, vid_dict["file_names"][i])
                for i in range(vid_dict["length"])
            ],
            "height": vid_dict["height"],
            "width": vid_dict["width"],
            "length": vid_dict["length"]
        }
        
        # for flickr30k entity
        if has_caption:
            record["caption"] = vid_dict['caption']
            # only exit in flickr30k entity but not flickr30k
            record["tokens_positive_eval"] = vid_dict.get("tokens_positive_eval", None)
        if has_pan_mask:
            record["pan_seg_file_names"] = [
                os.path.join(pan_gt_root, vid_dict["file_names"][i].replace(".jpg", ".png"))
                for i in range(vid_dict["length"])
            ]

        video_id = record["video_id"] = vid_dict["id"]

        # for UAV123
        if "video" in vid_dict:
            record["video"] = vid_dict["video"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                if has_expression:
                    assert "expressions" in obj and "exp_id" in obj
                    # for ref-youtube-vos and ref-davis evaluation
                    # obj["expressions"] = anno["expressions"]
                    # obj["exp_id"] = anno["exp_id"]

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if has_mask and not has_pan_mask:
                    if not (_segm and _segm[frame_idx]):
                        continue
                else:
                    if not (_bboxes and _bboxes[frame_idx]):
                        continue

                if "ori_id" in anno:
                    # for VOS inference
                    obj["ori_id"] = anno["ori_id"]
                bbox = _bboxes[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if has_mask and not has_pan_mask:
                    segm = _segm[frame_idx]
                    if isinstance(segm, dict):
                        if isinstance(segm["counts"], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm["size"])
                    elif segm:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]  # should start from 1

                frame_objs.append(obj)
            video_objs.append(frame_objs)
        
        if has_expression:
            # refvos task for evaluation
            record["exp_id"] = [anno["exp_id"] for anno in anno_dict_list]
            record["expressions"] = [anno["expressions"] for anno in anno_dict_list]

        record["annotations"] = video_objs
        record["has_mask"] = has_mask
        record["has_caption"] = has_caption
        record["has_stuff"] = has_stuff
        if has_pan_mask:
            record["metadata"] = metadata
        # language-guided detection
        if has_expression:
            record["task"] = "grounding"
            record["dataset_name"] = dataset_name
        elif sot or "sa_1b" in dataset_name:
            record["task"] = "sot"
            record["dataset_name"] = "sot_" + dataset_name
        else:
            record["task"] = "detection"
            if dataset_name.startswith("objects365"):
                record["dataset_name"] = "objects365"  # 365 classes
            elif dataset_name.startswith("coco"):
                if "panoptic" in dataset_name:
                    record["dataset_name"] = "coco_panoptic"  # 133 classes
                else:
                    record["dataset_name"] = "coco"  # 80 classes
            elif dataset_name.startswith("entityseg"):
                if "panoptic" in dataset_name:
                    record["dataset_name"] = "entityseg_panoptic"
                else:
                    record["dataset_name"] = "entityseg_instance"
            elif dataset_name.startswith("lvis"):
                record["dataset_name"] = "lvis"  # 1023 classes
            elif dataset_name.startswith("ade20k"):
                record["dataset_name"] = dataset_name  # 150 classes
            elif dataset_name.startswith("ytvis_2019"):
                record["dataset_name"] = "ytvis19"  # 40 classes
            elif dataset_name.startswith("ytvis_2021"):
                record["dataset_name"] = "ytvis21"  # 40 classes
            elif dataset_name.startswith("ovis"):
                record["dataset_name"] = "ovis"  # 25 classes
            elif dataset_name.startswith("mots_bdd_box_track") \
                    or dataset_name.startswith("mots_bdd_seg_track"):
                record["dataset_name"] = "bdd_track"  # 8 classes
            elif dataset_name.startswith("mots_burst"):
                record["dataset_name"] = "burst"  # 482 classes (in lvis)
            elif dataset_name.startswith("vipseg"):
                record["dataset_name"] = "vipseg"  # 124 classes
            elif dataset_name.startswith("vspw"):
                record["dataset_name"] = "vspw"  # 124 classes (same with VIPSeg)
            elif dataset_name.startswith('flickr'):
                record["dataset_name"] = 'flickr'
            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name} ")
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
              "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_ytvis_instances(name, metadata, json_file, image_root, evaluator_type=None,
                             has_mask=True, has_expression=False, has_caption=False, sot=False,
                             pan_gt_root=None, has_stuff=False):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.
    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
        evaluator_type: the type of evaluator
        has_mask: has mask, boxes or expression
        has_expression: expression in grounding task
        has_caption: flickr30k entity
        sot: single object tracking or segmentation
        pan_gt_root: the ground-truth masks root for panoptic segmentation
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    if has_expression:
        extra_annotation_keys = ["expressions", "exp_id"]
    elif has_caption:
        extra_annotation_keys = ['tokens_positive']
    else:
        extra_annotation_keys = []

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name,
        lambda: load_ytvis_json(
            json_file, image_root, name, has_mask=has_mask, extra_annotation_keys=extra_annotation_keys,
            has_expression=has_expression, has_caption=has_caption, sot=sot, 
            pan_gt_root=pan_gt_root, has_stuff=has_stuff
        )
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type=evaluator_type,
        **metadata
    )


if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ytvis_2019_train")

    json_file = "./datasets/ytvis/instances_train_sub.json"
    image_root = "./datasets/ytvis/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "ytvis-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2]
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
