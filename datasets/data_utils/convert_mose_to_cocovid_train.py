import json
import argparse
import os
from PIL import Image
import numpy as np
import cv2
import pycocotools.mask as maskUtils
from detectron2.structures import PolygonMasks


def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--data_root", default="mose", type=str, help="directory of MOSE")
    parser.add_argument("--mask_format", default="rle", choices=["polygon", "rle"], type=str)
    return parser.parse_args()


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
    args = parse_args()

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    data_root = os.path.join(_root, args.data_root)

    splits = ["train"]
    for split in splits:
        assert split == "train"
        new_data = {"videos": [], "annotations": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
        data_dir = os.path.join(data_root, split)
        images_dir = os.path.join(data_dir, "JPEGImages")
        masks_dir = os.path.join(data_dir, "Annotations")
        vid_idx = 0
        inst_idx = 0
        video_names = next(os.walk(images_dir))[1]
        masks_names = next(os.walk(masks_dir))[1]
        print("%d videos are found in the %s split"%(len(video_names), split))

        n_objs_total = 0

        for _, vid in enumerate(video_names):
            vid = vid
            if vid not in masks_names:
                print(f'Skip {vid} ...')
                continue

            vid_idx += 1
            vid_dict = {"height": None, "width": None, "length": None, "file_names": None, "id": vid_idx}
            vid_img_dir = os.path.join(images_dir, vid)
            vid_mask_dir = os.path.join(masks_dir, vid)
            frames = sorted(os.listdir(vid_img_dir))
            masks = sorted(os.listdir(vid_mask_dir))

            if len(frames) <= 5:
                continue

            assert len(frames) == len(masks)
            vid_dict["length"] = len(frames)
            vid_dict["file_names"] = [os.path.join(vid, x) for x in frames]
            init_frame_path = os.path.join(images_dir, vid_dict["file_names"][0])
            H, W, _ = cv2.imread(init_frame_path).shape
            vid_dict["height"], vid_dict["width"] = H, W
            new_data["videos"].append(vid_dict)

            vid_obj_dict = {}
            for frame_idx in range(vid_dict["length"]):
                mask_path = os.path.join(vid_mask_dir, masks[frame_idx])
                mask = Image.open(mask_path).convert('P')
                mask = np.array(mask)
                obj_ids = (np.unique(mask)).tolist()
                H, W = mask.shape
                # loop over obj_id in a video
                for obj_id in obj_ids:
                    if obj_id == 0:
                        continue

                    # get annos
                    mask_cur = (mask == int(obj_id)).astype(np.uint8) # 0, 1 binary
                    # some frame didn't contain the instance
                    if (mask_cur > 0).any():
                        if obj_id not in vid_obj_dict:
                            vid_obj_dict[obj_id] = {
                                "video_id": vid_idx, "id": obj_id + n_objs_total, "iscrowd": 0, "category_id": 1,
                                "bboxes": [None]*len(frames), "areas": [None]*len(frames),
                                "segmentations": [None]*len(frames)
                            }

                        box = bounding_box(mask_cur)
                        area = int(box[-2] * box[-1])
                        vid_obj_dict[obj_id]["bboxes"][frame_idx] = box
                        vid_obj_dict[obj_id]["areas"][frame_idx] = area
                        if args.mask_format == "polygon":
                            vid_obj_dict[obj_id]["segmentations"][frame_idx] = mask2polygon(mask_cur)
                        elif args.mask_format == "rle":
                            vid_obj_dict[obj_id]["segmentations"][frame_idx] = mask2rle(mask_cur)
                        else:
                            raise ValueError("Unsupported mask format")

            n_objs_total += len(vid_obj_dict)

            # save to annotations
            for k, v in vid_obj_dict.items():
                new_data["annotations"].append(v)

            print("%d/%d complete"%(vid_idx, len(video_names)))

        output_json = os.path.join(data_dir, "%s.json"%split)
        json.dump(new_data, open(output_json, 'w'))
        
'''
--ytvos
    --train
        --JPEGImages
        --Annotations
    --val
    --meta.json
    --train.json (after convert)
'''
