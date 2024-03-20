import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)
from .ovis import _get_ovis_instances_meta
from .burst import _get_burst_meta

from .sa_1b import register_sa_1b_instances
from .lvis import _get_lvis_instances_meta
from .coco_panoptic import _get_coco_panoptic_metadata, register_all_coco_panoptic_annos_sem_seg
from .ade20k_panoptic import _get_ade20k_panoptic_metadata
from .vps import _get_vipseg_panoptic_metadata, _get_vipseg_panoptic_metadata_val
from .vss import _get_vspw_vss_metadata
from .viposeg import _get_viposeg_metadata_val
from .entityseg import _get_entityseg_instance_meta, _get_entityseg_panoptic_meta

# referring task
from .refcoco import _get_refcoco_meta, register_refcoco


_PREDEFINED_SPLITS_SA_1B = {
    "sa_1b_train_250k_1": ("sa_1b/images", "sa_1b/annotations_250k/annotations_250k_1.json"),
    "sa_1b_train_250k_2": ("sa_1b/images", "sa_1b/annotations_250k/annotations_250k_2.json"),
}


def register_all_sa_1b(root):
    metadata = dict()
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SA_1B.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_sa_1b_instances(
            key,
            metadata,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

# ==== Predefined splits for LVIS ====
_PREDEFINED_SPLITS_LVIS = {
    # only use 7089 images, whose smaller size is above 512p
    "lvis_v1_train512p": ("coco/", "lvis/lvis_v1_train_video_512p.json"),
    # 100k images, whose larger size is above 360
    "lvis_v1_train_video": ("coco/", "lvis/lvis_v1_train_video.json"),
}

def register_all_lvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVIS.items():
        register_ytvis_instances(
            key,
            _get_lvis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type="coco",
        )

# panoptic datasets:
# coco-pan
_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_panoptic_train": (
        "coco/train2017",
        "coco/annotations/panoptic_train2017_cocovid.json",
        "coco/panoptic_train2017",  # gt_root to store .PNF files for gt masks
    ),
}

# ade20k
_PREDEFINED_SPLITS_ADE20K_PANOPTIC = {
    "ade20k_panoptic_train": (
        "ADEChallengeData2016/images/training",
        "ADEChallengeData2016/ade20k_panoptic_train_cocovid.json",
        "ADEChallengeData2016/ade20k_panoptic_train",  # gt_root to store .PNF files for gt masks
    ),
}


def register_coco_panoptic_train(root):
    for key, (image_root, json_file, pan_seg_gt_root) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_coco_panoptic_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type="coco_panoptic_seg",
            pan_gt_root=os.path.join(root, pan_seg_gt_root),
            has_stuff=True,
        )


def register_all_ade20k_panoptic(root):
    for key, (image_root, json_file, pan_seg_gt_root) in _PREDEFINED_SPLITS_ADE20K_PANOPTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ade20k_panoptic_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type="ade20k_panoptic_seg",
            pan_gt_root=os.path.join(root, pan_seg_gt_root),
            has_stuff=True,
        )

# TODO: Chech the sizes between images and masks
_PREDEFINED_SPLITS_ENTITYSEG_INSTANCE = {
    "entityseg_instance_train":(
        "entityseg/images/",
        "entityseg/annotations/entityseg_insseg_train_cocovid.json",
    )
}

def register_all_entityseg_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ENTITYSEG_INSTANCE.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_entityseg_instance_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type='ytvis',
        )
    
_PREDEFINED_SPLITS_ENTITYSEG_PANOPTIC = {
    "entityseg_panoptic_train":(
        "entityseg/images/",
        "entityseg/annotations/entityseg_panseg_train_cocovid.json",
    )
}

def register_all_entityseg_panoptic(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ENTITYSEG_PANOPTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_entityseg_panoptic_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type='ytvis',
        )

_PREDEFINED_SPLITS_VIPSeg_PANOPTIC = {
    "vipseg_panoptic_train": (
        "vipseg/imgs",
        "vipseg/panoptic_gt_VIPSeg_train_cocovid.json",
        "vipseg/panomasksRGB",  # gt_root to store .PNF files for gt masks
    ),
}

_PREDEFINED_SPLITS_VIPSeg_PANOPTIC_VAL = {
    "vipseg_panoptic_val": (
        "vipseg/VIPSeg_720P/imgs",
        "vipseg/VIPSeg_720P/panoptic_gt_VIPSeg_val_cocovid.json",
        "vipseg/VIPSeg_720P/panomasksRGB",  # gt_root to store .PNF files for gt masks
    ),
    "vipseg_panoptic_dev": (
        "vipseg/VIPSeg_720P/imgs",
        "vipseg/VIPSeg_720P/panoptic_gt_VIPSeg_val_sub_cocovid.json",
        "vipseg/VIPSeg_720P/panomasksRGB",  # gt_root to store .PNF files for gt masks
    ),
    "vipseg_panoptic_dev0.01": (
        "vipseg/VIPSeg_720P/imgs",
        "vipseg/VIPSeg_720P/panoptic_gt_VIPSeg_val_sub0.01_cocovid.json",
        "vipseg/VIPSeg_720P/panomasksRGB",  # gt_root to store .PNF files for gt masks
    ),
}

def register_all_vipseg_panoptic(root):
    for key, (image_root, json_file, pan_seg_gt_root) in _PREDEFINED_SPLITS_VIPSeg_PANOPTIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_vipseg_panoptic_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type="video_panoptic_seg",
            pan_gt_root=os.path.join(root, pan_seg_gt_root),
            has_stuff=True,
        )
    
    for key, (image_root, json_file, pan_seg_gt_root) in _PREDEFINED_SPLITS_VIPSeg_PANOPTIC_VAL.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_vipseg_panoptic_metadata_val(json_file, pan_seg_gt_root),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type="video_panoptic_seg",
            pan_gt_root=os.path.join(root, pan_seg_gt_root),
            has_stuff=True,
        )

# ====    Predefined splits for OVIS    ===========
_PREDEFINED_SPLITS_VSPW = {
    "vspw_vss_video_val": (
        "VSPW_480p/data/",
        "VSPW_480p/val_cocovid.json",
    ),
    "vspw_vss_video_dev": (
        "VSPW_480p/data/",
        "VSPW_480p/dev_cocovid.json",  # first 50 videos in val set, only for debug
    ),
}

def register_all_vspw_semantic(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VSPW.items():
        split_txt = json_file.split('/')[-1].split('_')[0] + '.txt'

        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_vspw_vss_metadata(split_txt),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type='video_semantic_seg',
            has_stuff=True,
        )


# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train_sub.json"),  # 90% videos in training set
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid21.json"),
    "ytvis_2022_val": ("ytvis_2021/valid22/JPEGImages",
                       "ytvis_2021/valid22.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_dev0.01": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/valid_sub_0.01.json"),  # 10% videos in training set
    "ytvis_2021_dev": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/valid_sub.json"),  # 10% videos in training set
    "ytvis_2021_dev_merge": ("ytvis_2021/train/JPEGImages",
                             "ytvis_2021/valid_sub_merge_car_truck.json"),
    
}


# ====    Predefined splits for OVIS    ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train/JPEGImages",
                   "ovis/train_sub.json"),  # 90% videos in training set
    "ovis_dev": ("ovis/train/JPEGImages",
                 "ovis/valid_sub.json"),  # 10% videos in training set
    "ovis_dev0.01": ("ovis/train/JPEGImages",
                 "ovis/valid_sub_0.01.json"),  # 10% videos in training set
    "ovis_dev_merge": ("ovis/train/JPEGImages",
                       "ovis/valid_sub_merge_motorbike.json"),
    "ovis_val": ("ovis/valid/JPEGImages",
                 "ovis/valid.json"),
    "ovis_test": ("ovis/test/JPEGImages",
                  "ovis/test.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type='ytvis',
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type='ytvis',
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            evaluator_type='ytvis',
        )

_PREDEFINED_SPLITS_SOT = {
    #"sot_got10k_train": ("got10k/train", "got10k/train.json", "vos"),
    #"sot_got10k_val": ("got10k/val", "got10k/val.json", "vos"),
    #"sot_got10k_test": ("got10k/test", "got10k/test.json", "vos"),
    "sot_ytbvos18_train": ("ytbvos/train/JPEGImages", "ytbvos/train.json", "vos"),
    "sot_ytbvos18_val": ("ytbvos/valid/JPEGImages", "ytbvos/valid.json", "vos"),
    "sot_davis16_train": ("DAVIS/JPEGImages/Full-Resolution", "DAVIS/2016_train.json", "davis"),
    "sot_davis16_val": ("DAVIS/JPEGImages/Full-Resolution", "DAVIS/2016_val.json", "davis"),
    "sot_davis17_train": ("DAVIS/JPEGImages/Full-Resolution", "DAVIS/2017_train.json", "davis"),
    "sot_davis17_val": ("DAVIS/JPEGImages/Full-Resolution", "DAVIS/2017_val.json", "davis"),
    #"sot_nfs": ("nfs/sequences", "nfs/nfs.json", "vos"),
    #"sot_uav123": ("UAV123/data_seq/UAV123", "UAV123/UAV123.json", "vos"),
}

# only one class for visual grounding
SOT_CATEGORIES = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}]


def _get_sot_meta():
    thing_ids = [k["id"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_all_sot(root):
    for key, (image_root, json_file, evaluator_type) in _PREDEFINED_SPLITS_SOT.items():
        has_mask = ("coco" in key) or ("vos" in key) or ("davis" in key)
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=has_mask,
            sot=True,
            evaluator_type=evaluator_type, # "vos" or "davis"
        )

# ====    Predefined splits for MOSE    ===========
_PREDEFINED_SPLITS_MOSE = {
    "mots_mose_train": ("mose/train/JPEGImages", "mose/train/train.json"),
    "mots_mose_val": ("mose/valid/JPEGImages", "mose/valid/valid.json"),
    "mots_mose_dev": ("mose/valid/JPEGImages", "mose/valid/valid_sub.json"),
    "mots_mose_test": ("mose/test/JPEGImages", "mose/test/test.json"),
}

def register_all_mose(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_MOSE.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True,
            sot=True,
            evaluator_type="vos",
        )


# ==== Predefined splits for TAO/BURST  ===========
_PREDEFINED_SPLITS_BURST = {
    "mots_burst_train": ("burst/frames/train",
                         "burst/annotations/train_uni.json",
                         "vos"),
    "mots_burst_val_vos": ("burst/frames/val",
                          "burst/annotations/val_first_frame_uni.json",
                          "vos"),   # point, box, mask as visual prompts
    "mots_burst_val_det": ("burst/frames/val",
                           "burst/annotations/val_first_frame_uni.json",
                           "ytvis"), # category-guided common segmentation
}

def register_all_burst(root):
    for key, (image_root, json_file, evaluator) in _PREDEFINED_SPLITS_BURST.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_burst_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True,
            sot=evaluator=='vos',
            evaluator_type=evaluator,
        )

_PREDEFINED_SPLITS_VIPOSeg = {
    "pvos_viposeg_val": ("viposeg/valid/JPEGImages", "viposeg/valid/valid_cocovid.json"),
    "pvos_viposeg_dev": ("viposeg/valid/JPEGImages", "viposeg/valid/dev_cocovid.json"),
    "pvos_viposeg_dev0.25": ("viposeg/valid/JPEGImages", "viposeg/valid/dev0.25_cocovid.json"),
}

def register_all_viposeg(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VIPOSeg.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_viposeg_metadata_val(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True,
            sot=True,
            evaluator_type="pvos",
            has_stuff=True,
        )

# ====    Predefined splits for Referring YTVIS    ===========
_PREDEFINED_SPLITS_REFYTBVOS = {
    "rvos-refytb-train": ("ytbvos/train/JPEGImages", "ytbvos/train_ref.json", "vos"),
    "rvos-refytb-val": ("ytbvos/valid19/JPEGImages", "ytbvos/valid19_ref.json", "vos"),
    # unsupervised
    "rvos-refdavis-val-0": ("ref-davis/valid/JPEGImages", "ref-davis/valid_0.json", "davis"),
    "rvos-refdavis-val-1": ("ref-davis/valid/JPEGImages", "ref-davis/valid_1.json", "davis"),
    "rvos-refdavis-val-2": ("ref-davis/valid/JPEGImages", "ref-davis/valid_2.json", "davis"),
    "rvos-refdavis-val-3": ("ref-davis/valid/JPEGImages", "ref-davis/valid_3.json", "davis"),
}

def register_all_refytbvos_videos(root):
    for key, (image_root, json_file, evaluator_type) in _PREDEFINED_SPLITS_REFYTBVOS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=True,
            has_expression=True,
            evaluator_type=evaluator_type,  # "vos"
        )

# ==== Predefined splits for REFCOCO datasets ===========
_PREDEFINED_SPLITS_REFCOCO_TRAIN = {
    # mixed refcoco data with cocovid format, selected 55809 images and 321007 expressions
    "rvos_refcoco-mixed": ("coco/train2017", "refcoco/refcoco-mixed/instances_train_video360p.json", True, True, False),
    "flickr30k_entity-train": ("flickr30k/flickr30k-images", "flickr30k/mdetr/final_flickr_mergedGT_train_cocovid.json", False, False, True)
}

def register_refcoco_mixed_train(root):
    for key, (image_root, json_file, has_mask, has_expression, has_caption) in _PREDEFINED_SPLITS_REFCOCO_TRAIN.items():
        register_ytvis_instances(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_mask=has_mask,
            has_expression=has_expression,
            has_caption=has_caption,
        )

# ==== Predefined splits for REFCOCO datasets ===========
_PREDEFINED_SPLITS_REFCOCO = {
    # refcoco
    "refcoco-unc-train": ("coco/train2017", "refcoco/refcoco/instances_refcoco_train.json"),
    "refcoco-unc-val": ("coco/train2017", "refcoco/refcoco/instances_refcoco_val.json"),
    "refcoco-unc-testA": ("coco/train2017", "refcoco/refcoco/instances_refcoco_testA.json"),
    "refcoco-unc-testB": ("coco/train2017", "refcoco/refcoco/instances_refcoco_testB.json"),
    # refcocog
    "refcocog-umd-train": ("coco/train2017", "refcoco/refcocog/instances_refcocog_train.json"),
    "refcocog-umd-val": ("coco/train2017", "refcoco/refcocog/instances_refcocog_val.json"),
    "refcocog-umd-test": ("coco/train2017", "refcoco/refcocog/instances_refcocog_test.json"),
    # "refcocog-google-val": ("coco/train2017", "refcoco/refcocog-google/instances_refcocogg_val.json"),
    # refcoco+
    "refcocoplus-unc-train": ("coco/train2017", "refcoco/refcoco+/instances_refcoco+_train.json"),
    "refcocoplus-unc-val": ("coco/train2017", "refcoco/refcoco+/instances_refcoco+_val.json"),
    "refcocoplus-unc-testA": ("coco/train2017", "refcoco/refcoco+/instances_refcoco+_testA.json"),
    "refcocoplus-unc-testB": ("coco/train2017", "refcoco/refcoco+/instances_refcoco+_testB.json"),
}

def register_all_refcoco(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFCOCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_refcoco(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ovis(_root)
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_burst(_root)

    register_all_sa_1b(_root)
    register_all_lvis(_root)
    
    # entityseg
    register_all_entityseg_instance(_root)
    register_all_entityseg_panoptic(_root)
    # panoptic seg
    register_coco_panoptic_train(_root)  # cocovid for train
    #register_all_coco_panoptic_annos_sem_seg(_root)  # coco for eval
    register_all_ade20k_panoptic(_root)
    register_all_vipseg_panoptic(_root)
    register_all_vspw_semantic(_root)

    # SOT
    register_all_sot(_root)
    register_all_mose(_root)
    register_all_viposeg(_root)

    # R-VOS
    register_all_refytbvos_videos(_root)

    # refcoco-mixed only
    register_refcoco_mixed_train(_root)
    register_all_refcoco(_root)
