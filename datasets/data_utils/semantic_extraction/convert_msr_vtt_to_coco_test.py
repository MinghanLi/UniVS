# convert videos into COCO annotation format
import os
import csv
import json
import random
import argparse
import torch
import torchvision


def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--video_dir", default="datasets/msr-vtt/data/TestVideo", type=str, help="")
    parser.add_argument("--data_file", default="datasets/msr-vtt/data/test_videodatainfo.json", type=str, help="")
    parser.add_argument("--out_json", default="datasets/msr-vtt/data/test_cocovid.json", type=str, help="")
    parser.add_argument("--video_stride", default=1, type=int, help="the stride of videos")
    parser.add_argument("--frame_stride", default=5, type=int, help="the stride of video frames")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    src_data = json.load(open(args.data_file, 'r'))
    # dict_keys(['info', 'videos', 'sentences'])
    # 2990 videos: {'category': 10, 'url': 'https://www.youtube.com/watch?v=Sa4BUsvAcjc', 'video_id': 'video7010', 'start time': 401.51, 'end time': 419.65, 'split': 'test', 'id': 7010}
    # 59800 sentences: {'caption': 'a band performing in a small club', 'video_id': 'video7960', 'sen_id': 140200}

    categories = [{'id': 1, 'name': 'object', 'isthing': 1, 'color': [120, 120, 120]}]
    dataset = {'videos': [], 'categories': categories, 'annotations': []}

    dataset_frames = 0
    dataset_videos = 0
    # List all video_names in the directory
    # ./--07WQ2iBlw_000001_000011.mp4
    # ./--33Lscn6sk_000004_000014.mp4
    video_names = os.listdir(args.video_dir)
    video_ids = [video_name.split('/')[-1].split('.')[0] for video_name in video_names]

    # Loop through the video_names and read them
    src_videos = src_data['videos'][::args.video_stride]
    video_sen_pairs = {src_video['video_id']: [] for src_video in src_videos}
    for sen_i, sen in enumerate(src_data['sentences']):
        if sen['video_id'] in video_sen_pairs:
            video_sen_pairs[sen['video_id']].append(sen_i)

    for i, src_video in enumerate(src_videos):
        video_id = src_video['video_id']
        
        if video_id not in video_ids:
            print(f"Unfound video with video_id: {video_id}")
            continue

        if i % 10 == 0:
            print(f'Processing video {i+1}/{len(src_videos)}')
        
        video_name = video_id + '.mp4'
        vpath = os.path.join(args.video_dir, video_name)
        # read all frames in the video
        vframes, aframes, info = torchvision.io.read_video(
            filename=vpath, pts_unit="sec", output_format="TCHW"
        )
        total_frames = len(vframes)
        height, width = vframes.shape[-2:]
        
        dataset_frames += total_frames
        dataset_videos += 1
        
        # generated frame names with '000001.jpg'
        file_names = [
            os.path.join(video_name, ''.join((6-len(str(t))) * ['0']) + str(t)+'.jpg')
            for t in range(total_frames)[::args.frame_stride]
        ]
        vid_dict = {
            "length": len(file_names),
            "file_names": file_names,
            "width": width,
            "height": height,
            "id": video_id,
            "caption_ids": video_sen_pairs[video_id]
        }
        dataset["videos"].append(vid_dict)
        
    # save
    print('total_frames: ', dataset_frames, dataset_videos)
    out_json = args.out_json.replace(".json", f"_{args.video_stride}_video_stride_{args.frame_stride}_frame_stride.json")
    print('Saving data into ', out_json)
    with open(out_json, "w") as f:
        json.dump(dataset, f)