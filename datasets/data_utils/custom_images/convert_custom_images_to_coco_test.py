# convert videos into COCO annotation format
import os
import json
import torch
import torchvision
import argparse
import cv2
import glob

def parse_args():
    parser = argparse.ArgumentParser("image to coco annotations")
    parser.add_argument("--video_dir", default="datasets/custom_images/raw", type=str, help="")
    parser.add_argument("--out_json", default="datasets/custom_images/raw/test.json", type=str, help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    categories = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}]
    dataset = {'videos': [], 'categories': categories, 'annotations': []}

    dataset_frames = 0
    # List all files in the directory
    # ./-nsrVzXLUBc_00:03:02.667_00:03:12.667.mp4
    # ./-oeq_F7Z7Io_00:01:06.833_00:01:16.833.mp4
    files = os.listdir(args.video_dir)
    print('Number of videos:', len(files))
    # Loop through the files and read them
    for i, video_name in enumerate(files):
        if i >= 1000:
            break

        if i % 100 == 0:
            print(f'Processing video {i+1}/{len(files)}')
        
        vpath = os.path.join(args.video_dir, video_name)
        if os.path.isdir(vpath):
            file_names = os.listdir(vpath)
            image = cv2.imread(os.path.join(vpath, file_names[0]))
            height, width = image.shape[:-1]
            file_names = [
                os.path.join(video_name, file_name) 
                for file_name in file_names 
                if file_name.split(".")[-1] in ("jpg", "png")
            ]
            total_frames = len(file_names)
            vid_dict = {
                "length": total_frames,
                "file_names": file_names,
                "width": width,
                "height": height,
                "id": video_name
            }

        else:
            print(f"{vpath} is not a dir! Skip...")
        
        dataset["videos"].append(vid_dict)
        dataset_frames += total_frames
        
    # save
    print('total_frames:', dataset_frames)
    print('Saving data into', args.out_json)
    with open(args.out_json, "w") as f:
        json.dump(dataset, f)