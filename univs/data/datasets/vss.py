import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

def gen_video_vspw_lists(image_root, split_txt):
    with open(split_txt, 'r') as f:
        lines = f.readlines()
        v_list = [line[:-1] for line in lines]
    ret = []
    for video_name in v_list:
        path_video = os.path.join(image_root, video_name)
        img_files = os.listdir(os.path.join(path_video, 'origin'))
        img_files.sort()
        img_files = [os.path.join(path_video, 'origin', item) for item in img_files]
        if os.path.exists(os.path.join(path_video, 'mask')):
            mask_files = os.listdir(os.path.join(path_video, 'mask'))
            mask_files.sort()
            mask_files = [os.path.join(path_video, 'mask', item) for item in mask_files]
        else:
            mask_files = [None] * len(img_files)
        ret.append({'video_id': video_name,
                    'file_names': img_files,
                    'sem_mask_names': mask_files})
    assert len(ret), f"No videos found in {image_root}!"
    return ret

def register_video_vspw_vss(
        name, metadata, image_root, split_txt,
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    DatasetCatalog.register(
        name,
        lambda: gen_video_vspw_lists(
            image_root, split_txt
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        evaluator_type="video_semantic_seg",
        ignore_label=255,
        **metadata,
    )


def _get_vspw_vss_metadata(split_txt=None):
    '''
    image_root: "datasets/VSPW_480p/data", help='/your/path/to/VSPW_480p'
    split: 'train', 'val', 'dev' or 'test'
    '''
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.

    classes = [k["name"] for k in CATEGORIES]
    colors = [k["color"] for k in CATEGORIES]

    meta["stuff_classes"] = classes
    meta["stuff_colors"] = colors
    meta["thing_classes"] = None
    meta["thing_colors"] = None

    classes_id = [k['id'] for k in CATEGORIES]
    meta['stuff_classes_id'] = classes_id
    meta['thing_classes_id'] = None

    dataset_id_to_contiguous_id = {}
    for i, id_ in enumerate(classes_id):
        dataset_id_to_contiguous_id[id_] = i
    meta["stuff_dataset_id_to_contiguous_id"] = dataset_id_to_contiguous_id
    meta["thing_dataset_id_to_contiguous_id"] = None

    meta["ignore_label"] = 255
    meta["split_txt"] = split_txt

    return meta


# original data register for VSS datamapper. Not used in oour code, but keep it for easier understand.
# we convert vss into vis format for unified training in builtin.py
_PREDEFINED_SPLITS_PANOVSPW = {
    "VSPW_vss_video_train": (
        "VSPW_480p/data/",
        "VSPW_480p/train.txt",
    ),
    "VSPW_vss_video_val": (
        "VSPW_480p/data/",
        "VSPW_480p/val.txt",
    ),
    "VSPW_vss_video_test": (
        "VSPW_480p/data/",
        "VSPW_480p/test.txt",
    ),
}

def register_all_video_panoVSPW(root):
    for (prefix, (image_root, split_txt)) in _PREDEFINED_SPLITS_PANOVSPW.items():
        metadata = _get_vspw_vss_metadata(image_root, split_txt)
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_video_vspw_vss(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, split_txt),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_video_panoVSPW(_root)

# orginal category id starts from 1
CATEGORIES = [
    {'id': 1, 'name': 'wall', 'isthing': 0, 'color': [120, 120, 120]},
    {'id': 2, 'name': 'ceiling', 'isthing': 0, 'color': [180, 120, 120]},
    {'id': 3, 'name': 'door', 'isthing': 1, 'color': [6, 230, 230]},
    {'id': 4, 'name': 'stair', 'isthing': 0, 'color': [80, 50, 50]},
    {'id': 5, 'name': 'ladder', 'isthing': 1, 'color': [4, 200, 3]},
    {'id': 6, 'name': 'escalator', 'isthing': 0, 'color': [120, 120, 80]},
    {'id': 7, 'name': 'Playground_slide', 'isthing': 0, 'color': [140, 140, 140]},
    {'id': 8, 'name': 'handrail_or_fence', 'isthing': 0, 'color': [204, 5, 255]},
    {'id': 9, 'name': 'window', 'isthing': 1, 'color': [230, 230, 230]},
    {'id': 10, 'name': 'rail', 'isthing': 0, 'color': [4, 250, 7]},
    {'id': 11, 'name': 'goal', 'isthing': 1, 'color': [224, 5, 255]},
    {'id': 12, 'name': 'pillar', 'isthing': 0, 'color': [235, 255, 7]},
    {'id': 13, 'name': 'pole', 'isthing': 0, 'color': [150, 5, 61]},
    {'id': 14, 'name': 'floor', 'isthing': 0, 'color': [120, 120, 70]},
    {'id': 15, 'name': 'ground', 'isthing': 0, 'color': [8, 255, 51]},
    {'id': 16, 'name': 'grass', 'isthing': 0, 'color': [255, 6, 82]},
    {'id': 17, 'name': 'sand', 'isthing': 0, 'color': [143, 255, 140]},
    {'id': 18, 'name': 'athletic_field', 'isthing': 0, 'color': [204, 255, 4]},
    {'id': 19, 'name': 'road', 'isthing': 0, 'color': [255, 51, 7]},
    {'id': 20, 'name': 'path', 'isthing': 0, 'color': [204, 70, 3]},
    {'id': 21, 'name': 'crosswalk', 'isthing': 0, 'color': [0, 102, 200]},
    {'id': 22, 'name': 'building', 'isthing': 0, 'color': [61, 230, 250]},
    {'id': 23, 'name': 'house', 'isthing': 0, 'color': [255, 6, 51]},
    {'id': 24, 'name': 'bridge', 'isthing': 0, 'color': [11, 102, 255]},
    {'id': 25, 'name': 'tower', 'isthing': 0, 'color': [255, 7, 71]},
    {'id': 26, 'name': 'windmill', 'isthing': 0, 'color': [255, 9, 224]},
    {'id': 27, 'name': 'well_or_well_lid', 'isthing': 0, 'color': [9, 7, 230]},
    {'id': 28, 'name': 'other_construction', 'isthing': 0, 'color': [220, 220, 220]},
    {'id': 29, 'name': 'sky', 'isthing': 0, 'color': [255, 9, 92]},
    {'id': 30, 'name': 'mountain', 'isthing': 0, 'color': [112, 9, 255]},
    {'id': 31, 'name': 'stone', 'isthing': 0, 'color': [8, 255, 214]},
    {'id': 32, 'name': 'wood', 'isthing': 0, 'color': [7, 255, 224]},
    {'id': 33, 'name': 'ice', 'isthing': 0, 'color': [255, 184, 6]},
    {'id': 34, 'name': 'snowfield', 'isthing': 0, 'color': [10, 255, 71]},
    {'id': 35, 'name': 'grandstand', 'isthing': 0, 'color': [255, 41, 10]},
    {'id': 36, 'name': 'sea', 'isthing': 0, 'color': [7, 255, 255]},
    {'id': 37, 'name': 'river', 'isthing': 0, 'color': [224, 255, 8]},
    {'id': 38, 'name': 'lake', 'isthing': 0, 'color': [102, 8, 255]},
    {'id': 39, 'name': 'waterfall', 'isthing': 0, 'color': [255, 61, 6]},
    {'id': 40, 'name': 'water', 'isthing': 0, 'color': [255, 194, 7]},
    {'id': 41, 'name': 'billboard_or_Bulletin_Board', 'isthing': 0, 'color': [255, 122, 8]},
    {'id': 42, 'name': 'sculpture', 'isthing': 1, 'color': [0, 255, 20]},
    {'id': 43, 'name': 'pipeline', 'isthing': 0, 'color': [255, 8, 41]},
    {'id': 44, 'name': 'flag', 'isthing': 1, 'color': [255, 5, 153]},
    {'id': 45, 'name': 'parasol_or_umbrella', 'isthing': 1, 'color': [6, 51, 255]},
    {'id': 46, 'name': 'cushion_or_carpet', 'isthing': 0, 'color': [235, 12, 255]},
    {'id': 47, 'name': 'tent', 'isthing': 1, 'color': [160, 150, 20]},
    {'id': 48, 'name': 'roadblock', 'isthing': 1, 'color': [0, 163, 255]},
    {'id': 49, 'name': 'car', 'isthing': 1, 'color': [140, 140, 140]},
    {'id': 50, 'name': 'bus', 'isthing': 1, 'color': [250, 10, 15]},
    {'id': 51, 'name': 'truck', 'isthing': 1, 'color': [20, 255, 0]},
    {'id': 52, 'name': 'bicycle', 'isthing': 1, 'color': [31, 255, 0]},
    {'id': 53, 'name': 'motorcycle', 'isthing': 1, 'color': [255, 31, 0]},
    {'id': 54, 'name': 'wheeled_machine', 'isthing': 0, 'color': [255, 224, 0]},
    {'id': 55, 'name': 'ship_or_boat', 'isthing': 1, 'color': [153, 255, 0]},
    {'id': 56, 'name': 'raft', 'isthing': 1, 'color': [0, 0, 255]},
    {'id': 57, 'name': 'airplane', 'isthing': 1, 'color': [255, 71, 0]},
    {'id': 58, 'name': 'tyre', 'isthing': 0, 'color': [0, 235, 255]},
    {'id': 59, 'name': 'traffic_light', 'isthing': 0, 'color': [0, 173, 255]},
    {'id': 60, 'name': 'lamp', 'isthing': 0, 'color': [31, 0, 255]},
    {'id': 61, 'name': 'person', 'isthing': 1, 'color': [11, 200, 200]},
    {'id': 62, 'name': 'cat', 'isthing': 1, 'color': [255, 82, 0]},
    {'id': 63, 'name': 'dog', 'isthing': 1, 'color': [0, 255, 245]},
    {'id': 64, 'name': 'horse', 'isthing': 1, 'color': [0, 61, 255]},
    {'id': 65, 'name': 'cattle', 'isthing': 1, 'color': [0, 255, 112]},
    {'id': 66, 'name': 'other_animal', 'isthing': 1, 'color': [0, 255, 133]},
    {'id': 67, 'name': 'tree', 'isthing': 0, 'color': [255, 0, 0]},
    {'id': 68, 'name': 'flower', 'isthing': 0, 'color': [255, 163, 0]},
    {'id': 69, 'name': 'other_plant', 'isthing': 0, 'color': [255, 102, 0]},
    {'id': 70, 'name': 'toy', 'isthing': 0, 'color': [194, 255, 0]},
    {'id': 71, 'name': 'ball_net', 'isthing': 0, 'color': [0, 143, 255]},
    {'id': 72, 'name': 'backboard', 'isthing': 0, 'color': [51, 255, 0]},
    {'id': 73, 'name': 'skateboard', 'isthing': 1, 'color': [0, 82, 255]},
    {'id': 74, 'name': 'bat', 'isthing': 0, 'color': [0, 255, 41]},
    {'id': 75, 'name': 'ball', 'isthing': 1, 'color': [0, 255, 173]},
    {'id': 76, 'name': 'cupboard_or_showcase_or_storage_rack', 'isthing': 0, 'color': [10, 0, 255]},
    {'id': 77, 'name': 'box', 'isthing': 1, 'color': [173, 255, 0]},
    {'id': 78, 'name': 'traveling_case_or_trolley_case', 'isthing': 1, 'color': [0, 255, 153]},
    {'id': 79, 'name': 'basket', 'isthing': 1, 'color': [255, 92, 0]},
    {'id': 80, 'name': 'bag_or_package', 'isthing': 1, 'color': [255, 0, 255]},
    {'id': 81, 'name': 'trash_can', 'isthing': 0, 'color': [255, 0, 245]},
    {'id': 82, 'name': 'cage', 'isthing': 0, 'color': [255, 0, 102]},
    {'id': 83, 'name': 'plate', 'isthing': 1, 'color': [255, 173, 0]},
    {'id': 84, 'name': 'tub_or_bowl_or_pot', 'isthing': 1, 'color': [255, 0, 20]},
    {'id': 85, 'name': 'bottle_or_cup', 'isthing': 1, 'color': [255, 184, 184]},
    {'id': 86, 'name': 'barrel', 'isthing': 1, 'color': [0, 31, 255]},
    {'id': 87, 'name': 'fishbowl', 'isthing': 1, 'color': [0, 255, 61]},
    {'id': 88, 'name': 'bed', 'isthing': 1, 'color': [0, 71, 255]},
    {'id': 89, 'name': 'pillow', 'isthing': 1, 'color': [255, 0, 204]},
    {'id': 90, 'name': 'table_or_desk', 'isthing': 1, 'color': [0, 255, 194]},
    {'id': 91, 'name': 'chair_or_seat', 'isthing': 1, 'color': [0, 255, 82]},
    {'id': 92, 'name': 'bench', 'isthing': 1, 'color': [0, 10, 255]},
    {'id': 93, 'name': 'sofa', 'isthing': 1, 'color': [0, 112, 255]},
    {'id': 94, 'name': 'shelf', 'isthing': 0, 'color': [51, 0, 255]},
    {'id': 95, 'name': 'bathtub', 'isthing': 0, 'color': [0, 194, 255]},
    {'id': 96, 'name': 'gun', 'isthing': 1, 'color': [0, 122, 255]},
    {'id': 97, 'name': 'commode', 'isthing': 1, 'color': [0, 255, 163]},
    {'id': 98, 'name': 'roaster', 'isthing': 1, 'color': [255, 153, 0]},
    {'id': 99, 'name': 'other_machine', 'isthing': 0, 'color': [0, 255, 10]},
    {'id': 100, 'name': 'refrigerator', 'isthing': 1, 'color': [255, 112, 0]},
    {'id': 101, 'name': 'washing_machine', 'isthing': 1, 'color': [143, 255, 0]},
    {'id': 102, 'name': 'Microwave_oven', 'isthing': 1, 'color': [82, 0, 255]},
    {'id': 103, 'name': 'fan', 'isthing': 1, 'color': [163, 255, 0]},
    {'id': 104, 'name': 'curtain', 'isthing': 0, 'color': [255, 235, 0]},
    {'id': 105, 'name': 'textiles', 'isthing': 0, 'color': [8, 184, 170]},
    {'id': 106, 'name': 'clothes', 'isthing': 0, 'color': [133, 0, 255]},
    {'id': 107, 'name': 'painting_or_poster', 'isthing': 1, 'color': [0, 255, 92]},
    {'id': 108, 'name': 'mirror', 'isthing': 1, 'color': [184, 0, 255]},
    {'id': 109, 'name': 'flower_pot_or_vase', 'isthing': 1, 'color': [255, 0, 31]},
    {'id': 110, 'name': 'clock', 'isthing': 1, 'color': [0, 184, 255]},
    {'id': 111, 'name': 'book', 'isthing': 0, 'color': [0, 214, 255]},
    {'id': 112, 'name': 'tool', 'isthing': 0, 'color': [255, 0, 112]},
    {'id': 113, 'name': 'blackboard', 'isthing': 0, 'color': [92, 255, 0]},
    {'id': 114, 'name': 'tissue', 'isthing': 0, 'color': [0, 224, 255]},
    {'id': 115, 'name': 'screen_or_television', 'isthing': 1, 'color': [112, 224, 255]},
    {'id': 116, 'name': 'computer', 'isthing': 1, 'color': [70, 184, 160]},
    {'id': 117, 'name': 'printer', 'isthing': 1, 'color': [163, 0, 255]},
    {'id': 118, 'name': 'Mobile_phone', 'isthing': 1, 'color': [153, 0, 255]},
    {'id': 119, 'name': 'keyboard', 'isthing': 1, 'color': [71, 255, 0]},
    {'id': 120, 'name': 'other_electronic_product', 'isthing': 0, 'color': [255, 0, 163]},
    {'id': 121, 'name': 'fruit', 'isthing': 0, 'color': [255, 204, 0]},
    {'id': 122, 'name': 'food', 'isthing': 0, 'color': [255, 0, 143]},
    {'id': 123, 'name': 'instrument', 'isthing': 1, 'color': [0, 255, 235]},
    {'id': 124, 'name': 'train', 'isthing': 1, 'color': [133, 255, 0]},
]