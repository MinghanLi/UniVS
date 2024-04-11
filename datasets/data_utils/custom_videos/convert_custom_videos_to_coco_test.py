# convert videos into COCO annotation format
import os
import json
import torch
import torchvision
import argparse
import cv2
import glob

def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--video_dir", default="datasets/custom_videos/raw", type=str, help="")
    parser.add_argument("--out_json", default="datasets/custom_videos/raw.json", type=str, help="")
    parser.add_argument("--is_256p", default=False, type=bool, help="compressed videos with 256p (short edge)")
    return parser.parse_args()


def save_frames_into_a_video(height, width, frame_dir, output_file=None):
    if output_file is None:
        output_file = frame_dir + '.avi'

    # save all frames to a .avi video
    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*'DIVX'),
        4,
        (width, height)
    )

    file_names = glob.glob('/'.join([frame_dir, "*.jpg"]))
    file_names = sorted(
        file_names, 
        key=lambda f: int(f.split("/")[-1].split('_')[-1].replace('.jpg', ''))
    )
    for file_name in file_names:
        out.write(cv2.imread(file_name))
    out.release()
    print(f"save all frames with .jpg frames into {output_file}")


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
        
        path = os.path.join(args.video_dir, video_name)
        if os.path.isdir(path):
            file_names = os.listdir(path)
            total_frames = len(file_names)
            image = cv2.imread(os.path.join(path, file_names[0]))
            height, width = image.shape[:-1]
            file_names = [
                os.path.join(video_name, file_name) 
                for file_name in file_names 
                if file_name.split(".")[-1] in ("jpg", "png")
            ]
            vid_dict = {
                "length": total_frames,
                "file_names": file_names,
                "width": width,
                "height": height,
                "id": video_name
            }

        else:
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
                "id": ext,
            }
        
        dataset["videos"].append(vid_dict)
        dataset_frames += total_frames
        
    # save
    print('total_frames:', dataset_frames)
    print('Saving data into', args.out_json)
    with open(args.out_json, "w") as f:
        json.dump(dataset, f)