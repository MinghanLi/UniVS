import json
import os
import cv2
import torch
import numpy as np

from PIL import Image
from detectron2.structures import PolygonMasks
import pycocotools.mask as maskUtils

def compute_area(segmentation):
    if isinstance(segmentation, list):
        polygons = PolygonMasks([segmentation])
        area = polygons.area()[0].item()
    elif isinstance(segmentation, dict):  # RLE
        area = maskUtils.area(segmentation).item()
    else:
        raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
    return area

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # (x1, y1, w, h) 

def mask2polygon(input_mask):
    contours, hierarchy = cv2.findContours(input_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        result.append(segmentation)
    return result

def mask2rle(input_mask):
    rle = maskUtils.encode(np.array(input_mask, order="F", dtype="uint8"))
    if not isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


if __name__ == "__main__":
    _root = 'datasets/viposeg/valid/'
    json_file = 'meta.json'
    obj_cls_file = 'obj_class.json'

    _seq_list_file = os.path.join(_root, json_file)
    if not os.path.isfile(_seq_list_file):
        print('Not find file:', _seq_list_file)
    else:
        ann_f = json.load(open(_seq_list_file, 'r'))['videos']
    
    obj_categories = json.load(open(os.path.join(_root, obj_cls_file), 'r'))
                
    seqs = list(ann_f.keys())
    image_root = os.path.join(_root, 'JPEGImages')
    label_root = os.path.join(_root, 'Annotations')
    obj_class_file = os.path.join(_root,'obj_class.json')
    thing_class = [2, 4, 8, 10, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 60, 61, 62, 63,
                64, 65, 72, 74, 76, 77, 78, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 99, 100,
                101, 102, 106, 107, 108, 109, 114, 115, 116, 117, 118, 122, 123]
    stuff_class = [0, 1, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 53, 57, 58, 59, 66, 67, 68, 69,
                70, 71, 73, 75, 80, 81, 93, 94, 98, 103, 104, 105, 110, 111, 112, 113, 119, 120, 121]

    is_rgb_img = False
    mask_format = "rle"
    splits = ['valid', 'dev', 'dev0.25']
    for split in splits:
        new_data = {"videos": [], "annotations": [], "categories": [{"supercategory": "object", "id": 1, "name": "object"}]}
        if split in {'valid'}:
            num_videos = len(seqs)
        elif split in {'dev'}:
            num_videos = int(len(seqs) * 0.1)
        elif split in {'dev0.25'}:
            num_videos = int(len(seqs) * 0.25)
        
        for vid_idx in range(num_videos):
            seq_name = seqs[vid_idx]
            data = ann_f[seq_name]['objects']
            obj_names = list(data.keys())
            images = []
            labels = []
            for obj_n in obj_names:
                images += map(lambda x: x + '.jpg', list(data[obj_n]["frames"]))
                labels.append(data[obj_n]["frames"][0] + '.png')
            images = sorted(list(set(images)))
            labels = sorted(list(set(labels)))
            category_ids = obj_categories[seq_name]

            img_path = os.path.join(image_root, seq_name, images[0])
            img = np.array(cv2.imread(img_path), dtype=np.float32)
        
            vid_len = len(images)
            vid_dict = {
                "id": seq_name,
                "length": vid_len,
                "width": img.shape[1],
                "height": img.shape[0],
                "file_names": [os.path.join(seq_name, img_name) for img_name in images],
            }
            new_data["videos"].append(vid_dict)
            
            vid_obj_dict = {}
            for obj_id in obj_names:
                vid_obj_dict[int(obj_id)] = {
                    "video_id": seq_name, 
                    "id": int(obj_id), 
                    "iscrowd": 0, 
                    "category_id": category_ids[obj_id], 
                    "bboxes": [None]*vid_len, 
                    "segmentations": [None]*vid_len, 
                    "areas": [None]*vid_len
                }
        
            for label_name in labels:
                label_path = os.path.join(label_root, seq_name, label_name)
                label = Image.open(label_path)
                label = np.array(label, dtype=np.uint8)
                exit_obj_ids = np.unique(label)
                
                frame_idx = images.index(label_name.replace('.png', '.jpg'))
                
                for obj_id in exit_obj_ids:
                    if obj_id not in vid_obj_dict.keys():
                        continue
                        
                    # get annos
                    mask_cur = (label==int(obj_id)).astype(np.uint8) # 0,1 binary
                    # some frame didn't contain the instance
                    if (mask_cur > 0).any():
                        box = bounding_box(mask_cur)
                        area = int(np.sum(label==int(obj_id)))
                    
                    vid_obj_dict[obj_id]["bboxes"][frame_idx] = box
                    if mask_format == "polygon":
                        vid_obj_dict[obj_id]["segmentations"][frame_idx] = mask2polygon(mask_cur)
                    elif mask_format == "rle":
                        vid_obj_dict[obj_id]["segmentations"][frame_idx] = mask2rle(mask_cur)
                    else:
                        raise ValueError("Unsupported mask format")
                    vid_obj_dict[obj_id]["areas"][frame_idx] = area
            
            # save to annotations
            for k, v in vid_obj_dict.items():
                new_data["annotations"].append(v)
            print("%d/%d complete"%(vid_idx, len(seqs)))

        output_json = os.path.join(_root, "%s_cocovid.json"%split)
        print('Saving annotations in:', output_json)
        json.dump(new_data, open(output_json, 'w'))