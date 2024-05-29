# convert videos into COCO annotation format
import os
import json
import csv
import torch
import torchvision
import argparse

from decord import VideoReader
from decord import cpu, gpu


def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--subset_index", default=0, type=int, help="index of subset data")
    parser.add_argument("--csv_file", default="datasets/pexel/PexelVideos-6s-Parts/PexelVideos-6s_index.csv", type=str, help="")
    parser.add_argument("--video_dir", default="datasets/pexel/PexelVideos-Full", type=str, help="")
    parser.add_argument("--out_json", default="datasets/pexel/json_files_cocovid/PexelVideos-6s_index_cocovid.json", type=str, help="")
    parser.add_argument("--start_index", default=-1, type=int, help="start index")
    parser.add_argument("--end_index", default=10000, type=int, help="end index")
    parser.add_argument("--is_specified_indices", default=False, type=bool, help="extract specified videos")
    parser.add_argument("--specified_indices", default=[55115, 55116, 90377, 90378, 90379, 90380, 90381, 96101], 
                        type=list, help="specified indices")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.csv_file = args.csv_file.replace("_index", "_"+str(args.subset_index))
    args.out_json = args.out_json.replace("_index", "_"+str(args.subset_index))
    print("Input file is: ", args.csv_file)

    categories = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}]
    dataset = {'videos': [], 'categories': categories, 'annotations': []}

    skip_header = True
    with open(args.csv_file, "r") as f:
        reader = csv.reader(f)
        # Skip the first row (header)
        if skip_header:
            next(reader, None)
        if args.is_specified_indices:
            print(f'Processing videos with index {args.specified_indices}....')
            reader = list(reader)
            samples = [reader[idx] for idx in args.specified_indices]
        else:
            samples = list(reader)
            if args.start_index >= 0:
                print(f'Processing videos from {args.start_index} to {args.end_index}....')
                samples = samples[args.start_index:args.end_index]

    dataset_frames = 0
    # List all files in the directory
    # ./121142413.hd.mp4_00:00:00.079_00:00:06.079
    print('Number of videos:', len(samples))
    # Loop through the files and read them
    for i, (video_name_start_end_time, caption) in enumerate(samples):
        video_name, start_time, end_time = video_name_start_end_time.split('_')
        video_id = video_name.replace('.mp4', '')  # 121142413.hd.mp4
        start_time = start_time.split('.')[0]  # 00:00:00.079 => 00:00:00
        end_time = end_time.split('.')[0]      # 00:00:06.079 => 00:00:06
        start_sec = int(start_time.split(':')[0])*3600 + int(start_time.split(':')[1]) * 60 + int(start_time.split(':')[2])
        end_sec = int(end_time.split(':')[0])*3600 + int(end_time.split(':')[1]) * 60 + int(end_time.split(':')[2])

        if i % 50 == 0:
            print(f'Processing video {i+1}/{len(samples)}: {video_name}')
            
        ext = video_name.split(".")[-1]
        if ext.lower() not in ("mp4", "avi", "mov", "mkv"):
            print(f"Unsupported video format: {ext}")
            continue
                
        vpath = os.path.join(args.video_dir, video_name)
        try:
            # # read all frames in the video
            # vframes, aframes, info = torchvision.io.read_video(
            #     filename=vpath, start_pts=start_time, end_pts=end_time, pts_unit="sec", output_format="TCHW"
            # )
            # height, width = vframes.shape[-2:]
            # total_frames = len(vframes)

            # Open the video file
            vr = VideoReader(vpath, ctx=cpu(0))  # Use gpu(0) for GPU
            # Access video properties
            # Get the shape of the first frame to determine dimensions
            frame = vr[0]
            height, width, channels = frame.shape
            # Get video framerate to calculate frame numbers
            framerate = vr.get_avg_fps()
            # Calculate frame indices
            start_frame = int(start_sec * framerate)
            end_frame = int(end_sec * framerate)
            total_frames = end_frame - start_frame
            # Read frames from start to end
            # vframes = vr.get_batch(range(start_frame, end_frame)) 
            
        except Exception as e:
            print(f"An error occurred while loading the video {video_name}:", str(e))
            continue

        dataset_frames += total_frames

        # generated frame names with '000001.jpg'
        file_names = [
            video_name + '/' + ''.join((6-len(str(t))) * ['0']) + str(t)+'.jpg'
            for t in range(total_frames)
        ]
        vid_dict = {
            "length": total_frames,
            "file_names": file_names,
            "width": width,
            "height": height,
            "id": video_name_start_end_time,
            "framerate": framerate,
            "start_sec": start_sec,
            "end_sec": end_sec,
        }
        dataset["videos"].append(vid_dict)
        
    # save
    num_videos = len(dataset["videos"])
    print(f"total videos: {num_videos} and frames: {dataset_frames}")
    if args.is_specified_indices:
        out_json = args.out_json.replace('.json', f'_specified_videos.json')
    elif args.start_index >= 0:
        out_json = args.out_json.replace('.json', f'_{args.start_index}_{args.end_index}.json')
    else:
        out_json = args.out_json
    print('Saving data into', out_json)
    with open(out_json, "w") as f:
        json.dump(dataset, f)