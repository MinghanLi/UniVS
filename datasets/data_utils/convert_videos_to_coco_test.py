# convert videos into COCO annotation format
import os
import json
import torch
import torchvision
import argparse

def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--video_dir", default="datasets/internvid/raw/InternVId-FLT_1", type=str, help="")
    parser.add_argument("--out_json", default="datasets/internvid/raw/InternVId-FLT_1.json", type=str, help="")
    parser.add_argument("--is_256p", default=True, type=bool, help="compressed videos with 256p (short edge)")
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
            "id": video_name.replace('.mp4', '')
        }
        dataset["videos"].append(vid_dict)
        
    # save
    print('total_frames:', dataset_frames)
    print('Saving data into', args.out_json)
    with open(args.out_json, "w") as f:
        json.dump(dataset, f)