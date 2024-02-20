import logging

"""
This file contains functions to parse OVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

OVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Person"},
    {"color": [255, 109, 65], "isthing": 1, "id": 2, "name": "Bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 3, "name": "Cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 4, "name": "Dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 5, "name": "Horse"},
    {"color": [255, 208, 186], "isthing": 1, "id": 6, "name": "Sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "Cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 8, "name": "Elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 9, "name": "Bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 10, "name": "Zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 11, "name": "Giraffe"},
    {"color": [107, 142, 35], "isthing": 1, "id": 12, "name": "Poultry"},
    {"color": [0, 82, 0], "isthing": 1, "id": 13, "name": "Giant_panda"},
    {"color": [119, 11, 32], "isthing": 1, "id": 14, "name": "Lizard"},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": "Parrot"},
    {"color": [0, 60, 100], "isthing": 1, "id": 16, "name": "Monkey"},
    {"color": [100, 170, 30], "isthing": 1, "id": 17, "name": "Rabbit"},
    {"color": [166, 196, 102], "isthing": 1, "id": 18, "name": "Tiger"},
    {"color": [73, 77, 174], "isthing": 1, "id": 19, "name": "Fish"},
    {"color": [0, 143, 149], "isthing": 1, "id": 20, "name": "Turtle"},
    {"color": [134, 134, 103], "isthing": 1, "id": 21, "name": "Bicycle"},
    {"color": [0, 0, 230], "isthing": 1, "id": 22, "name": "Motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 23, "name": "Airplane"},
    {"color": [0, 0, 192], "isthing": 1, "id": 24, "name": "Boat"},
    {"color": [0, 0, 142], "isthing": 1, "id": 25, "name": "Vehical"},
]


def _get_ovis_instances_meta():
    thing_ids = [k["id"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 25, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret