#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from PIL import Image
import pycocotools.mask as maskUtils
from detectron2.structures import PolygonMasks

def compute_area(segmentation):
    if isinstance(segmentation, list):
        polygons = PolygonMasks([segmentation])
        area = polygons.area()[0].item()
    elif isinstance(segmentation, dict):  # RLE
        area = maskUtils.area(segmentation).item()
    else:
        raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
    return area

def convert_to_binary_mask(segm):
    if isinstance(segm, dict):  # RLE
        segm1 = maskUtils.decode(segm)
    elif isinstance(segm, list):
        # filter out invalid polygons (< 3 points)
        segm1 = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]

    return segm1.shape

if __name__ == "__main__":
    dataset_dir = os.getenv("DETECTRON2_DATASETS", "datasets")
    src_json = os.path.join(dataset_dir, f"entityseg/annotations/entityseg_insseg_train.json")
    des_json = os.path.join(dataset_dir, f"entityseg/annotations/entityseg_insseg_train_cocovid.json")

    src_dataset = json.load(open(src_json, 'r'))
    des_dataset = {'videos': [], 'categories': [], 'annotations': []}
    des_dataset["categories"] = src_dataset["categories"]
    
    image_shape_dict, image_file_dict = {}, {}
    original_images = len(src_dataset["images"])
    print(f'Number of images: {original_images}...')
    # videos with [h, w], where min(h, w) > 512, (remain 7089 images)
    for i, img_dict in enumerate(src_dataset["images"][100:]):
        if (i % int(0.1 * original_images)) == 0:
            print(f'processing {i * 10} of {original_images} images')
        
        vid_dict = {
            "length": 1,
            "file_names": [img_dict["file_name"]],
            "width": img_dict["width"],
            "height": img_dict["height"],
            "id": img_dict["id"]
        }
        des_dataset["videos"].append(vid_dict)

        image_shape_dict[img_dict["id"]] = [img_dict["height"], img_dict["width"]]
        image_file_dict[img_dict["id"]] = img_dict["file_name"]

    print(f'Number of annotationss: {len(src_dataset["annotations"])}...')
    # annotations
    for obj_dict in src_dataset["annotations"]:
        segm = obj_dict["segmentation"]

        # check image and mask sizes
        if obj_dict["image_id"] in image_shape_dict:
            # image size
            # img = Image.open(os.path.join(dataset_dir, 'entityseg/images', image_file_dict[obj_dict["image_id"]]))
            # mask size
            height_mask, width_mask = segm['size']
            if [height_mask, width_mask] != image_shape_dict[obj_dict["image_id"]]:
                print(
                    'image and mask sizes of dict:', image_shape_dict[obj_dict["image_id"]], height_mask, width_mask, 
                )
                raise ValueError('uninconsisitent shapes between annotation dict and images')

        area = compute_area(segm)
        anno_dict_new = {
            "video_id": obj_dict["image_id"],
            "iscrowd": obj_dict["iscrowd"],
            "length": 1,
            "id": obj_dict["id"],
            "category_id": obj_dict["category_id"],
            "bboxes": [obj_dict["bbox"]],
            "segmentations": [segm],
            "areas": [area]
        }

        des_dataset["annotations"].append(anno_dict_new)
        exit()
    num_annos = len(des_dataset["annotations"])
    print(f'Save {num_annos} thing/stuff annotations')

    # save
    with open(des_json, "w") as f:
        json.dump(des_dataset, f)