import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--src_dir", default="burst/annotations/val/first_frame_annotations.json", type=str,
                        help="")
    parser.add_argument("--des_json", default="burst/annotations/val_first_frame_uni.json",
                        type=str, help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _root = 'datasets'
    src_dataset = json.load(open(os.path.join(_root, args.src_dir), 'r'))
    n_videos = len(src_dataset["sequences"])

    # videos
    videos_dict = []
    annotations_dict = []
    for i, seq in enumerate(src_dataset["sequences"]):
        video_id = seq["id"]

        image_paths = [os.path.join(seq['dataset'], seq['seq_name'], p) for p in seq['annotated_image_paths']]
        video_len = len(image_paths)

        videos_dict.append({"length": video_len, "file_names": image_paths, \
                            "width": seq["width"], "height": seq["height"], "id": seq["id"]})

        seg_per_video = {}
        box_per_video = {}
        point_per_video = {}
        for fid, frame_annos in enumerate(seq['segmentations']):
            if len(frame_annos) == 0:
                continue
                
            for obj_id, obj_anno in frame_annos.items():
                if obj_id not in seg_per_video:
                    seg_per_video[obj_id] = [None] * video_len
                    box_per_video[obj_id] = [None] * video_len
                    point_per_video[obj_id] = [None] * video_len

                seg_per_video[obj_id][fid] = {'size': [seq["height"], seq["width"]], 'counts': obj_anno['rle']}
                box_per_video[obj_id][fid] = obj_anno['bbox']
                point_per_video[obj_id][fid] = obj_anno['point']

        for obj_id in seg_per_video.keys():
            assert len(seg_per_video[obj_id]) == video_len, \
                'annotations should have same number of frames as the video.'

            annotations_dict.append({
                "width": seq["width"], "height": seq["height"],
                "iscrowd": 0, "category_id": seq['track_category_ids'][obj_id], "id": obj_id,
                "video_id": video_id, "bboxes": box_per_video[obj_id], "areas": [None] * video_len,
                "segmentations": seg_per_video[obj_id], "points": point_per_video[obj_id]
            })

        print(f"Coverting {i} of {n_videos}")

    des_dataset = {'videos': videos_dict, 'categories': src_dataset['categories'],
                   'annotations': annotations_dict}

    # save
    with open(os.path.join(_root, args.des_json), "w") as f:
        json.dump(des_dataset, f)