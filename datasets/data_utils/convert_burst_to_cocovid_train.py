import os
import numpy as np
import json
import argparse

import pycocotools.mask as maskUtils


def parse_args():
    parser = argparse.ArgumentParser("image to video converter for TAO / burst")
    parser.add_argument("--src_dir", default="datasets/burst/annotations/train/train.json", type=str,
                        help="")
    parser.add_argument("--des_json", default="datasets/burst/annotations/train_uni.json",
                        type=str, help="")
    return parser.parse_args()


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # (x1, y1, w, h)


if __name__ == "__main__":
    args = parse_args()
    _root_image = 'datasets/burst/frames/train/'
    dataset_names = next(os.walk(_root_image))[1]

    src_dataset = json.load(open(args.src_dir, 'r'))
    n_videos = len(src_dataset["sequences"])

    obj_id_total = 0

    # videos
    videos_dict = []
    annotations_dict = []
    for i, seq in enumerate(src_dataset["sequences"]):
        video_id = seq["id"]

        if seq['dataset'] not in dataset_names:
            print('Does not find images, skip', seq['dataset'])
            continue

        image_paths = [os.path.join(seq['dataset'], seq['seq_name'], p) for p in seq['annotated_image_paths']]
        video_len = len(image_paths)

        videos_dict.append({"length": video_len, "file_names": image_paths, \
                            "width": seq["width"], "height": seq["height"],
                            "id": seq["id"]})

        objs_per_video = {}
        for fid, frame_annos in enumerate(seq['segmentations']):
            obj_ids_occur = []
            for obj_id, obj_seg in frame_annos.items():
                obj_ids_occur.append(obj_id)
                if obj_id not in objs_per_video:
                    objs_per_video[obj_id] = {
                        'bboxes': [None] * video_len,
                        'segms': [None] * video_len,
                        'areas': [None] * video_len
                    }

                segm = {'size': [seq["height"], seq["width"]], 'counts': obj_seg['rle']}

                objs_per_video[obj_id]['segms'][fid] = segm
                objs_per_video[obj_id]['bboxes'][fid] = maskUtils.toBbox(segm).tolist()
                objs_per_video[obj_id]['areas'][fid] = int(maskUtils.area(segm))

        for obj_id, obj_dict in objs_per_video.items():
            assert len(obj_dict['segms']) == video_len, 'annotations should have same number of frames as the video.'

            annotations_dict.append({
                "width": seq["width"], "height": seq["height"],
                "iscrowd": 0, "category_id": seq['track_category_ids'][obj_id], "id": obj_id_total+int(obj_id),
                "video_id": video_id, "bboxes": obj_dict['bboxes'], "areas": obj_dict['areas'],
                "segmentations": obj_dict['segms']
            })

        obj_id_total += len(objs_per_video)

        print(f"Coverting {i} of {n_videos}")

    des_dataset = {'videos': videos_dict, 'categories': src_dataset['categories'],
                   'annotations': annotations_dict}

    # save
    with open(args.des_json, "w") as f:
        json.dump(des_dataset, f)