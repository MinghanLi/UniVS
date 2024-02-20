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
    parser.add_argument("--data_dir", default="datasets/ytbvos", type=str, help="directory of ref-youtube-vos")
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
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)]  # (x1, y1, w, h)


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
    min_vid_len = 5 # there must be at least 2 frames in a video. Or it will be invalid.
    args = parse_args()
    data_dir = args.data_dir
    splits = ["train"]
    for split in splits:
        assert split == "train"
        new_data = {
            "videos": [],
            "annotations": [],
            "categories": [{"supercategory": "object", "id": 1, "name": "object"}]
        }

        inst_idx = 0
        # read object information
        img_folder = os.path.join(data_dir, split)
        with open(os.path.join(img_folder, 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        # read expression data
        ann_file = os.path.join(data_dir, "meta_expressions/%s/meta_expressions.json"%split)
        with open(ann_file, 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        assert len(subset_metas_by_video) == len(subset_expressions_by_video)
        videos = list(subset_expressions_by_video.keys())
        num_vids = len(videos)
        images_dir = os.path.join(data_dir, split, "JPEGImages")
        masks_dir = os.path.join(data_dir, split, "Annotations")

        exp_id = 0
        for vid_idx, vid in enumerate(videos):
            vid_img_dir = os.path.join(images_dir, vid)
            vid_mask_dir = os.path.join(masks_dir, vid)
            frames = sorted(os.listdir(vid_img_dir))
            masks = sorted(os.listdir(vid_mask_dir))
            assert len(frames) == len(masks)
            video_len = len(frames)
            init_frame_path = os.path.join(images_dir, vid, frames[0])
            H, W, _ = cv2.imread(init_frame_path).shape
            # parse video information
            file_names = [os.path.join(vid, frame_name) for frame_name in frames]
            vid_dict = {"height": H, "width": W, "length": video_len, "file_names": file_names, "id": vid_idx}
            new_data["videos"].append(vid_dict)

            # parse expressions in a video
            data_dict = {}
            for _, exp_data in subset_expressions_by_video[vid]["expressions"].items():
                exp, obj_id = exp_data["exp"], exp_data["obj_id"]
                if obj_id not in data_dict:
                    data_dict[obj_id] = {"exp": [], "frames": None}
                data_dict[obj_id]["exp"].append(exp)
            metas_vid = subset_metas_by_video[vid]["objects"]

            # save expressions to vid_dict
            vid_obj_dict = {}
            # parse mask information in the current video, one video can contain multiple objects
            for frame_idx in range(video_len):
                mask_path = os.path.join(vid_mask_dir, masks[frame_idx])
                mask = Image.open(mask_path).convert('P')
                mask = np.array(mask)

                H, W = mask.shape
                # loop over obj_id in a video
                for obj_id in metas_vid.keys():
                    if obj_id not in vid_obj_dict:
                        vid_obj_dict[obj_id] = {
                            "video_id": vid_idx,
                            "id": int(obj_id),
                            "iscrowd": 0,
                            "category_id": 1,
                            "bboxes": [None] * video_len,
                            "segmentations": [None] * video_len,
                            "areas": [None] * video_len,
                            "expressions": data_dict[obj_id]["exp"]
                        }

                    # get annos
                    mask_cur = (mask == int(obj_id)).astype(np.uint8)  # 0,1 binary
                    # some frame didn't contain the instance
                    if (mask_cur > 0).any():
                        box = bounding_box(mask_cur)
                        area = int(box[-2] * box[-1])
                        vid_obj_dict[obj_id]["bboxes"][frame_idx] = box
                        if args.mask_format == "polygon":
                            vid_obj_dict[obj_id]["segmentations"][frame_idx] = mask2polygon(mask_cur)
                        elif args.mask_format == "rle":
                            vid_obj_dict[obj_id]["segmentations"][frame_idx] = mask2rle(mask_cur)
                        else:
                            raise ValueError("Unsupported mask format")
                        vid_obj_dict[obj_id]["areas"][frame_idx] = area

            # save to annotations
            for obji_id, obj_dict in vid_obj_dict.items():
                valid_len = sum([1 for box in obj_dict["bboxes"] if box is not None])
                if valid_len >= min_vid_len:
                    # save annotation for per exp
                    for exp in obj_dict["expressions"]:
                        obj_dict_per_exp = {
                            "exp_id": exp_id,
                            "expressions": [exp]
                        }
                        for k, v in obj_dict.items():
                            if k not in obj_dict_per_exp:
                                obj_dict_per_exp[k] = v
                        exp_id += 1
                        # save
                        new_data["annotations"].append(obj_dict_per_exp)

            if vid_idx % int(0.01 * num_vids) == 0:
                print("%05d/%05d done."%(vid_idx+1, num_vids))

        output_json = os.path.join(data_dir, "%s_ref.json"%split)
        json.dump(new_data, open(output_json, 'w'))
        

'''
each instance corresponds a video with only one instance(annotation)
ytbvos 
--datasets
  --ytbvos
    --train
      --JPEEGImages
      --Annotations
      --meta.json
    --valid
    --meta_expressions
      --train/valid/test
        --meta_expressions.json
    --train.json 
    --valid.json 
    --train_ref.json (after convert)
    --valid_ref.json (after convert)
'''

