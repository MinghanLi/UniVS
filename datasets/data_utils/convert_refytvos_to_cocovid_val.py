"""
Each expression is a sample.
"""
import json
import argparse
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--data_dir", default="datasets/ytbvos", type=str, help="directory of ref-youtube-vos")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    split = "valid"
    new_data = {"videos": [], "annotations": [], "categories": [{"supercategory": "object", "id": 1, "name": "object"}]}

    img_folder = os.path.join(data_dir, split)
    # remove test set from valid_test set
    with open(os.path.join(data_dir, "meta_expressions", split, "meta_expressions.json"), 'r') as f:
        data = json.load(f)['videos']
        valid_test_videos = set(data.keys())
    with open(os.path.join(data_dir, "meta_expressions", "test", "meta_expressions.json"), 'r') as f:
        test_videos = set(json.load(f)['videos'].keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    assert len(video_list) == 202, 'error: incorrect number of validation videos'

    # 1. For each video
    video_id = 0
    inst_id = 0
    for video in tqdm(video_list):
        video_len = len(data[video]["frames"])
        frames = [os.path.join(video, x+".jpg") for x in data[video]["frames"]]
        H, W = cv2.imread(os.path.join(img_folder, "JPEGImages", frames[0])).shape[:-1]
        new_data["videos"].append({
            "height": H, "width": W, "length": video_len, "file_names": frames, "id": video_id
        })

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        # read all the anno meta
        for i in range(num_expressions):
            new_data["annotations"].append({
                "video_id": video_id, 
                "id": inst_id, 
                "category_id": 1,
                "exp_id": expression_list[i], 
                "expressions": [expressions[expression_list[i]]["exp"]]
            })

            inst_id += 1

        video_id += 1

    output_json = os.path.join(data_dir, "%s_ref.json"%split)
    json.dump(new_data, open(output_json, 'w'))
