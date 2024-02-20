# Copyright (c) Facebook, Inc. and its affiliates.
import logging

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)


REFCOCO_CATEGORIES = COCO_CATEGORIES


def _get_refcoco_meta():
    thing_ids = [k["id"] for k in REFCOCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in REFCOCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in REFCOCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


