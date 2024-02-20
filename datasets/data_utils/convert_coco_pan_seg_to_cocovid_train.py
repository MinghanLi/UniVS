#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

if __name__ == "__main__":
    dataset_dir = os.getenv("DETECTRON2_DATASETS", "datasets")
    src_json = os.path.join(dataset_dir, f"coco/annotations/panoptic_train2017.json")
    des_json = os.path.join(dataset_dir, f"coco/annotations/panoptic_train2017_cocovid.json")

    src_dataset = json.load(open(src_json, 'r'))
    des_dataset = {'videos': [], 'categories': [], 'annotations': []}
    des_dataset["categories"] = src_dataset["categories"]

    original_images = len(src_dataset["images"])

    included_images = []
    # videos with [h, w], where min(h, w) > 512, (remain 7089 images)
    for i, img_dict in enumerate(src_dataset["images"]):
        if (i % int(0.1 * original_images)) == 0:
            print(f'processing {i * 10} of {original_images} images')
        if max(img_dict["width"], img_dict["height"]) < 480:
            continue

        included_images.append(img_dict["id"])

        vid_dict = {
            "length": 1,
            "file_names": [img_dict["file_name"]],
            "width": img_dict["width"],
            "height": img_dict["height"],
            "id": img_dict["id"]
        }
        des_dataset["videos"].append(vid_dict)

    num_images = len(included_images)
    print(f'Select {num_images} images of {original_images}...')

    # annotations
    for anno_dict in src_dataset["annotations"]:
        if anno_dict['image_id'] not in included_images:
            continue

        for obj_dict in anno_dict["segments_info"]:
            anno_dict_new = {
                "video_id": anno_dict["image_id"],
                "iscrowd": obj_dict["iscrowd"],
                "length": 1,
                "id": obj_dict["id"],
                "category_id": obj_dict["category_id"],
                "bboxes": [obj_dict["bbox"]],
                "areas": [obj_dict["area"]]
            }

            des_dataset["annotations"].append(anno_dict_new)

    num_annos = len(des_dataset["annotations"])
    print(f'Save {num_annos} thing/stuff annotations')

    # save
    with open(des_json, "w") as f:
        json.dump(des_dataset, f)