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
    src_jsons = [
        os.path.join(dataset_dir, f"entityseg/annotations/entityseg_train_01.json"),
        os.path.join(dataset_dir, f"entityseg/annotations/entityseg_train_02.json"),
        os.path.join(dataset_dir, f"entityseg/annotations/entityseg_train_03.json"),
    ]
    des_json = os.path.join(dataset_dir, f"entityseg/annotations/entityseg_panseg_train_cocovid.json")

    des_dataset = {'videos': [], 'categories': [], 'annotations': []}
    acc_image_id = 0
    for src_json in src_jsons:
        print()
        print(f"Processing {src_json} .................")
        image_shape_dict, image_file_dict = {}, {}
        # map image ids form three datasets
        map_image_ids = {}

        src_dataset = json.load(open(src_json, 'r'))
        des_dataset["categories"] = src_dataset["categories"]
        original_images = len(src_dataset["images"])
        print(f'Number of images: {original_images}...')
        # videos with [h, w], where min(h, w) > 512, (remain 7089 images)
        for i, img_dict in enumerate(src_dataset["images"]):
            if (i % int(0.1 * original_images)) == 0:
                print(f'processing {i} of {original_images} images')
            
            # map image ids form three datasets
            if img_dict["id"] not in map_image_ids:
                map_image_ids[img_dict["id"]] = acc_image_id
                acc_image_id += 1
            image_id = map_image_ids[img_dict["id"]]

            vid_dict = {
                "length": 1,
                "file_names": [img_dict["file_name"]],
                "width": img_dict["width"],
                "height": img_dict["height"],
                "id": image_id
            }
            des_dataset["videos"].append(vid_dict)

            image_shape_dict[image_id] = [img_dict["height"], img_dict["width"]]
            image_file_dict[image_id] = img_dict["file_name"]

        print(f'Number of annotationss: {len(src_dataset["annotations"])}...')
        # annotations
        for i, obj_dict in enumerate(src_dataset["annotations"]):
            if (i % int(0.1 * len(src_dataset["annotations"]))) == 0:
                print(f'processing {i} of {len(src_dataset["annotations"])} annotations')

            segm = obj_dict["segmentation"]
            image_id = map_image_ids[obj_dict["image_id"]]
            # check image and mask sizes
            if image_id in image_shape_dict:
                # mask size
                height_mask, width_mask = segm['size']
                assert [height_mask, width_mask] == image_shape_dict[image_id], \
                    f"mismatched sizes between image and mask: {image_shape_dict[image_id]} and{(height_mask, width_mask)}"

                # if [height_mask, width_mask] != [img.size[1], img.size[0]]:
                #     # image size
                #     img = Image.open(os.path.join(dataset_dir, 'entityseg/images', image_file_dict[image_id]))
                #     print(
                #         'image and mask sizes of dict:', image_shape_dict[image_id], (height_mask, width_mask), 
                #         'loaded image size:', (img.size[1], img.size[0])
                #     )
                #     raise ValueError('uninconsisitent shapes between annotation dict and images')
  
            area = compute_area(segm)
            anno_dict_new = {
                "video_id": image_id,
                "iscrowd": obj_dict["iscrowd"],
                "length": 1,
                "id": obj_dict["id"],
                "category_id": obj_dict["category_id"],
                "bboxes": [obj_dict["bbox"]],
                "segmentations": [segm],
                "areas": [area]
            }

            des_dataset["annotations"].append(anno_dict_new)
    
    num_imgs = len(des_dataset["videos"])
    num_annos = len(des_dataset["annotations"])
    print(f'Save {num_imgs} images and {num_annos} thing/stuff annotations')

    # save
    with open(des_json, "w") as f:
        json.dump(des_dataset, f)