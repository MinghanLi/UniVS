import os
import numpy as np
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser("down-sampling frames with interval 5 in burst")
    parser.add_argument("--src_json", default="datasets/burst/annotations/train_uni.json", type=str,
                        help="")
    parser.add_argument("--des_json", default="datasets/burst/annotations/train_uni_itv5frames.json",
                        type=str, help="")
    parser.add_argument("--itv_frames", default=5, type=int, help="the interval of frames")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = json.load(open(args.src_json, 'r'))

    new_data = {"categories": data["categories"], "videos": [], "annotations": []}

    print("processing videos ...")
    for vid_dict in data["videos"]:
        new_vid_dict = {k: v for k, v in vid_dict.items() if k != "file_names"}
        new_vid_dict["file_names"] = vid_dict["file_names"][::args.itv_frames]
        new_vid_dict["length"] = len(new_vid_dict["file_names"])

        new_data["videos"].append(new_vid_dict)

    print("processing annotations...")
    for ann_dict in data["annotations"]:
        tgt_keys = {"bboxes", "areas", "segmentations"}
        new_ann_dict = {}
        for k, v in ann_dict.items():
            if k not in tgt_keys:
                new_ann_dict[k] = v
            else:
                new_ann_dict[k] = v[::args.itv_frames]
        new_data["annotations"].append(new_ann_dict)

    # save
    with open(args.des_json, "w") as f:
        json.dump(new_data, f)


