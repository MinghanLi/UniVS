# convert videos into COCO annotation format
import os
import json
import csv
import torch
import torchvision
import argparse

def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--csv_file", default="datasets/internvid/csv_files/InternVId-FLT_1_len5.csv", type=str, help="")
    parser.add_argument("--video_dir", default="datasets/internvid/raw/InternVId-FLT_1", type=str, help="")
    parser.add_argument("--out_json", default="datasets/internvid/csv_files_cocovid/InternVId-FLT_1_len5.json", type=str, help="")
    parser.add_argument("--start_index", default=0, type=int, help="start index")
    parser.add_argument("--end_index", default=10000, type=int, help="end index")
    parser.add_argument("--is_specified_indices", default=False, type=bool, help="extract specified videos")
    parser.add_argument("--specified_indices", default=[55115, 55116, 90377, 90378, 90379, 90380, 90381, 96101], 
                        type=list, help="specified indices")
    parser.add_argument("--is_256p", default=True, type=bool, help="compressed videos with 256p (short edge)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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
            print(f'Processing videos from {args.start_index} to {args.end_index}....')
            samples = list(reader)[args.start_index:args.end_index]

    dataset_frames = 0
    # List all files in the directory
    # ./-nsrVzXLUBc_00:03:02.667_00:03:12.667.mp4
    # ./-oeq_F7Z7Io_00:01:06.833_00:01:16.833.mp4
    print('Number of videos:', len(samples))
    # Loop through the files and read them
    for i, (video_name, caption) in enumerate(samples):
        if i % 500 == 0:
            print(f'Processing video {i+1}/{len(samples)}')
            
        ext = video_name.split(".")[-1]
        if ext.lower() not in ("mp4", "avi", "mov", "mkv"):
            print(f"Unsupported video format: {ext}")
            continue
                
        path = os.path.join(args.video_dir, video_name)
        # read all frames in the video
        vframes, aframes, info = torchvision.io.read_video(
            filename=path, pts_unit="sec", output_format="TCHW"
        )
        total_frames = len(vframes)
        height, width = vframes.shape[-2:]
        
        dataset_frames += total_frames
        
        # generated frame names with '000001.jpg'
        file_names = [
            os.path.join(video_name, ''.join((6-len(str(t))) * ['0']) + str(t)+'.jpg')
            for t in range(total_frames)
        ]
        
        vid_dict = {
            "length": total_frames,
            "file_names": file_names,
            "width": 2*width if args.is_256p else width,
            "height": 2*height if args.is_256p else height,
            "id": video_name.replace('.mp4', ''),
        }
        dataset["videos"].append(vid_dict)
        
    # save
    print('total_frames:', dataset_frames)
    if args.is_specified_indices:
        out_json = args.out_json.replace('.json', f'_specified_videos.json')
    else:
        out_json = args.out_json.replace('.json', f'_{args.start_index}_{args.end_index}.json')
    print('Saving data into', out_json)
    with open(out_json, "w") as f:
        json.dump(dataset, f)