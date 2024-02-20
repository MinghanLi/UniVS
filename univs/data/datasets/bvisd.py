import logging

"""
# ----------------------------- BVISD ------------------------------------
# 1. merge similar classes in three datasets, for example:
#    a) car, truck, bus -> vehicle. "Motorcycle" & 23 "Bicycle" -> Motorbike
#       A Bus is in a class by itself- a heavy vehicle for carrying multiple passengers.
#       A truck is a vehicle for carrying cargo and a bare minimum crew.
#       And a car is a light four wheelers for carrying upto 7 people (I am stretching the definition to include MUVs).
#       A vehicle is defined as a device for transporting individuals or objects
#       that can be powered by machinery, by a person, or by animals.
#   b) OVIS: duck <-> Poultry
#   c) Remove categories: {'n02419796': 'antelope'}, {'n02129165': 'lion'} in ImageNet VID
# 2. deal with the classes that do not annotated, for example
#   a) "Sheep" in ytvis2021
"""

logger = logging.getLogger(__name__)

COCO_TO_BVISD = {
    1:26, 2:23, 3:5, 4:23, 5:1, 6:5, 7:36, 8:37, 9:4, 16:3,
    17:6, 18:9, 19:19, 20:40, 21:7, 22:12, 23:2, 24:39, 25:18, 34:14,
    35:31, 36:31, 41:29, 42:33, 43:34
}

# YT21: 5 car & 37 truck -> 5 vehicle, 10 duck -> 10 Poultry
YTVIS_2021_TO_BVISD = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10,
    11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20,
    21:21, 22:22, 23:23, 24:24, 25:25, 26:26, 27:27, 28:28, 29:29, 30:30,
    31:31, 32:32, 33:33, 34:34, 35:35, 36:36, 37:5, 38:37, 39:38, 40:39,
}

BVISD_TO_YTVIS_2021 = {
    1:1, 2:2, 3:3, 4:4, 5:[5, 37], 6:6, 7:7, 8:8, 9:9, 10:10,
    11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 20:20,
    21:21, 22:22, 23:23, 24:24, 25:25, 26:26, 27:27, 28:28, 29:29, 30:30,
    31:31, 32:32, 33:33, 34:34, 35:35, 36:36, 37:38, 38:39, 39:40,
}

# ovis: 12 Poultry -> 10 duck
# 21 "Motorcycle" & 22 "Bicycle" -> 23 Motorbike
OVIS_TO_BVISD = {
    1:26, 2:3, 3:6, 4:9, 5:19, 6:40, 7:7, 8:12, 9:2, 10:39,
    11:18, 12:10, 13:17, 14:21, 15:25, 16:22, 17:27, 18:35, 19:13, 20:37,
    21:23, 22:23, 23:1, 24:4, 25:5
}

BVISD_TO_OVIS = {
    26:1, 3:2, 6:3, 9:4, 19:5, 40:6, 7:7, 12:8, 2:9, 39:10,
    18:11, 10:12, 17:13, 21:14, 25:15, 22:16, 27:17, 35:18, 13:19, 37:20,
    23:[21, 22], 1:23, 4:24, 5:25
}

BVISD_CATEGORIES = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "bear"},
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "boat"},  # watercraft
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "vehical"},  # OVIS: Vehical, YT21: car, truck, VID: bus, car,
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "cow"},  # cattle
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "deer"},
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "dog"},
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "poultry"},  # Poultry
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
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "motorbike"},  # "Motorcycle"
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "mouse"},  # 'hamster'
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
    {"color": [0, 143, 149], "isthing": 1, "id": 37, "name": "turtle"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "whale"},
    {"color": [199, 100, 0], "isthing": 1, "id": 39, "name": "zebra"},
    {"color": [255, 208, 186], "isthing": 1, "id": 40, "name": "sheep"},
]


def _get_bvisd_instances_meta():
    thing_ids = [k["id"] for k in BVISD_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in BVISD_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 40]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in BVISD_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret