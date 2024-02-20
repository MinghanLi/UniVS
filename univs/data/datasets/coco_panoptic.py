import os
import json

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager

# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_PANOPTIC_CATEGORIES = [
    {'supercategory': 'person', 'isthing': 1, 'id': 1, 'name': 'person'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 2, 'name': 'bicycle'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 3, 'name': 'car'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 4, 'name': 'motorcycle'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 5, 'name': 'airplane'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 6, 'name': 'bus'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 7, 'name': 'train'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 8, 'name': 'truck'},
    {'supercategory': 'vehicle', 'isthing': 1, 'id': 9, 'name': 'boat'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 10, 'name': 'traffic light'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 11, 'name': 'fire hydrant'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 13, 'name': 'stop sign'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 14, 'name': 'parking meter'},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 15, 'name': 'bench'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 16, 'name': 'bird'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 17, 'name': 'cat'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 18, 'name': 'dog'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 19, 'name': 'horse'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 20, 'name': 'sheep'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 21, 'name': 'cow'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 22, 'name': 'elephant'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 23, 'name': 'bear'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 24, 'name': 'zebra'},
    {'supercategory': 'animal', 'isthing': 1, 'id': 25, 'name': 'giraffe'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 27, 'name': 'backpack'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 28, 'name': 'umbrella'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 31, 'name': 'handbag'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 32, 'name': 'tie'},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 33, 'name': 'suitcase'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 34, 'name': 'frisbee'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 35, 'name': 'skis'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 36, 'name': 'snowboard'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 37, 'name': 'sports ball'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 38, 'name': 'kite'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 39, 'name': 'baseball bat'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 40, 'name': 'baseball glove'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 41, 'name': 'skateboard'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 42, 'name': 'surfboard'},
    {'supercategory': 'sports', 'isthing': 1, 'id': 43, 'name': 'tennis racket'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 44, 'name': 'bottle'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 46, 'name': 'wine glass'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 47, 'name': 'cup'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 48, 'name': 'fork'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 49, 'name': 'knife'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 50, 'name': 'spoon'},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 51, 'name': 'bowl'},
    {'supercategory': 'food', 'isthing': 1, 'id': 52, 'name': 'banana'},
    {'supercategory': 'food', 'isthing': 1, 'id': 53, 'name': 'apple'},
    {'supercategory': 'food', 'isthing': 1, 'id': 54, 'name': 'sandwich'},
    {'supercategory': 'food', 'isthing': 1, 'id': 55, 'name': 'orange'},
    {'supercategory': 'food', 'isthing': 1, 'id': 56, 'name': 'broccoli'},
    {'supercategory': 'food', 'isthing': 1, 'id': 57, 'name': 'carrot'},
    {'supercategory': 'food', 'isthing': 1, 'id': 58, 'name': 'hot dog'},
    {'supercategory': 'food', 'isthing': 1, 'id': 59, 'name': 'pizza'},
    {'supercategory': 'food', 'isthing': 1, 'id': 60, 'name': 'donut'},
    {'supercategory': 'food', 'isthing': 1, 'id': 61, 'name': 'cake'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 62, 'name': 'chair'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 63, 'name': 'couch'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 64, 'name': 'potted plant'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 65, 'name': 'bed'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 67, 'name': 'dining table'},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 70, 'name': 'toilet'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 72, 'name': 'tv'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 73, 'name': 'laptop'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 74, 'name': 'mouse'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 75, 'name': 'remote'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 76, 'name': 'keyboard'},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 77, 'name': 'cell phone'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 78, 'name': 'microwave'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 79, 'name': 'oven'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 80, 'name': 'toaster'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 81, 'name': 'sink'},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 82, 'name': 'refrigerator'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 84, 'name': 'book'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 85, 'name': 'clock'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 86, 'name': 'vase'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 87, 'name': 'scissors'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 88, 'name': 'teddy bear'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 89, 'name': 'hair drier'},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 90, 'name': 'toothbrush'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 92, 'name': 'banner'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 93, 'name': 'blanket'},
    {'supercategory': 'building', 'isthing': 0, 'id': 95, 'name': 'bridge'},
    {'supercategory': 'raw-material', 'isthing': 0, 'id': 100, 'name': 'cardboard'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 107, 'name': 'counter'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 109, 'name': 'curtain'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 112, 'name': 'door-stuff'},
    {'supercategory': 'floor', 'isthing': 0, 'id': 118, 'name': 'floor-wood'},
    {'supercategory': 'plant', 'isthing': 0, 'id': 119, 'name': 'flower'},
    {'supercategory': 'food-stuff', 'isthing': 0, 'id': 122, 'name': 'fruit'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 125, 'name': 'gravel'},
    {'supercategory': 'building', 'isthing': 0, 'id': 128, 'name': 'house'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 130, 'name': 'light'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 133, 'name': 'mirror-stuff'},
    {'supercategory': 'structural', 'isthing': 0, 'id': 138, 'name': 'net'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 141, 'name': 'pillow'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 144, 'name': 'platform'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 145, 'name': 'playingfield'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 147, 'name': 'railroad'},
    {'supercategory': 'water', 'isthing': 0, 'id': 148, 'name': 'river'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 149, 'name': 'road'},
    {'supercategory': 'building', 'isthing': 0, 'id': 151, 'name': 'roof'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 154, 'name': 'sand'},
    {'supercategory': 'water', 'isthing': 0, 'id': 155, 'name': 'sea'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 156, 'name': 'shelf'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 159, 'name': 'snow'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 161, 'name': 'stairs'},
    {'supercategory': 'building', 'isthing': 0, 'id': 166, 'name': 'tent'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 168, 'name': 'towel'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 171, 'name': 'wall-brick'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 175, 'name': 'wall-stone'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 176, 'name': 'wall-tile'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 177, 'name': 'wall-wood'},
    {'supercategory': 'water', 'isthing': 0, 'id': 178, 'name': 'water-other'},
    {'supercategory': 'window', 'isthing': 0, 'id': 180, 'name': 'window-blind'},
    {'supercategory': 'window', 'isthing': 0, 'id': 181, 'name': 'window-other'},
    {'supercategory': 'plant', 'isthing': 0, 'id': 184, 'name': 'tree-merged'},
    {'supercategory': 'structural', 'isthing': 0, 'id': 185, 'name': 'fence-merged'},
    {'supercategory': 'ceiling', 'isthing': 0, 'id': 186, 'name': 'ceiling-merged'},
    {'supercategory': 'sky', 'isthing': 0, 'id': 187, 'name': 'sky-other-merged'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 188, 'name': 'cabinet-merged'},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 189, 'name': 'table-merged'},
    {'supercategory': 'floor', 'isthing': 0, 'id': 190, 'name': 'floor-other-merged'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 191, 'name': 'pavement-merged'},
    {'supercategory': 'solid', 'isthing': 0, 'id': 192, 'name': 'mountain-merged'},
    {'supercategory': 'plant', 'isthing': 0, 'id': 193, 'name': 'grass-merged'},
    {'supercategory': 'ground', 'isthing': 0, 'id': 194, 'name': 'dirt-merged'},
    {'supercategory': 'raw-material', 'isthing': 0, 'id': 195, 'name': 'paper-merged'},
    {'supercategory': 'food-stuff', 'isthing': 0, 'id': 196, 'name': 'food-other-merged'},
    {'supercategory': 'building', 'isthing': 0, 'id': 197, 'name': 'building-other-merged'},
    {'supercategory': 'solid', 'isthing': 0, 'id': 198, 'name': 'rock-merged'},
    {'supercategory': 'wall', 'isthing': 0, 'id': 199, 'name': 'wall-other-merged'},
    {'supercategory': 'textile', 'isthing': 0, 'id': 200, 'name': 'rug-merged'}
]


def _get_coco_panoptic_metadata():
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    all_colors = {k["id"]: k["color"] for k in COCO_CATEGORIES}

    thing_classes = [k["name"] for k in COCO_PANOPTIC_CATEGORIES if k["isthing"] == 1]
    thing_ids = [k["id"] for k in COCO_PANOPTIC_CATEGORIES if k["isthing"] == 1]
    thing_colors = [all_colors[k["id"]] for k in COCO_PANOPTIC_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}

    thing_stuff_classes = [k["name"] for k in COCO_PANOPTIC_CATEGORIES]
    thing_stuff_ids = [k["id"] for k in COCO_PANOPTIC_CATEGORIES]
    thing_stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_stuff_ids)}
    thing_stuff_colors = [all_colors[k["id"]] for k in COCO_PANOPTIC_CATEGORIES]

    meta = {
        "thing_dataset_id_to_contiguous_id": thing_stuff_dataset_id_to_contiguous_id,
        "thing_classes": thing_stuff_classes,
        "thing_colors": thing_stuff_colors,
        "thing_only_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_only_classes": thing_classes,
        "thing_only_colors": thing_colors,
    }

    return meta


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
}


def load_coco_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_coco_panoptic_annos_sem_seg(
    name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json
):
    panoptic_name = name
    if hasattr(MetadataCatalog.get(panoptic_name), "thing_classes"):
        delattr(MetadataCatalog.get(panoptic_name), "thing_classes")
    if hasattr(MetadataCatalog.get(panoptic_name), "thing_colors"):
        delattr(MetadataCatalog.get(panoptic_name), "thing_colors")
    MetadataCatalog.get(panoptic_name).set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
        # thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    )

    # the name is "coco_2017_train_panoptic_with_sem_seg" and "coco_2017_val_panoptic_with_sem_seg"
    semantic_name = name + "_with_sem_seg"
    DatasetCatalog.register(
        semantic_name,
        lambda: load_coco_panoptic_json(
            panoptic_json, image_root, panoptic_root, sem_seg_root, metadata
        ),
    )
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_coco_panoptic_annos_sem_seg(root):
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        register_coco_panoptic_annos_sem_seg(
            prefix,
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


