import os, sys
import json
import glob
import numpy as np
import PIL.Image as Image
from tqdm import trange
from panopticapi.utils import IdGenerator, save_json
import argparse

original_format_folder = 'datasets/vipseg/VIPSeg_720P/panomasks'
# folder to store panoptic PNGs
out_folder = 'datasets/vipseg/VIPSeg_720P/panomasksRGB'
out_file = 'datasets/vipseg/VIPSeg_720P/panoptic_gt_VIPSeg.json'
with open('datasets/vipseg/VIPSeg_720P/panoVIPSeg_categories.json', 'r') as f:
    CATEGORIES = json.load(f)


def panoptic_video_converter():
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    with open('datasets/vipseg/train.txt', 'r') as f:
        lines = f.readlines()
        v_train_list = [line[:-1] for line in lines]
    v_train_videos = []
    v_train_annotations = []

    with open('datasets/vipseg/val.txt', 'r') as f:
        lines = f.readlines()
        v_val_list = [line[:-1] for line in lines]
    v_val_videos = []
    v_val_annotations = []

    with open('datasets/vipseg/test.txt', 'r') as f:
        lines = f.readlines()
        v_test_list = [line[:-1] for line in lines]
    v_test_videos = []
    v_test_annotations = []

    categories = CATEGORIES
    categories_dict = {el['id']: el for el in CATEGORIES}

    video_id = -1
    for video in sorted(os.listdir(original_format_folder)):
        print('processing video:{}'.format(video))
        video_id += 1

        id_generator = IdGenerator(categories_dict)
        instid2color = {}

        vid_dict = dict()
        annotations = dict()
        vid_len = len(sorted(os.listdir(os.path.join(original_format_folder, video))))
        for i, image_filename in enumerate(sorted(os.listdir(os.path.join(original_format_folder, video)))):
            original_format = np.array(Image.open(os.path.join(original_format_folder, video, image_filename)))
            assert original_format.shape[0] == 720
            
            if i == 0:
                vid_dict = {
                    "id": video_id,
                    "length": vid_len,
                    "width": original_format.shape[1],
                    "height": original_format.shape[0],
                    "file_names": []
                }
            
            vid_dict["file_names"].append(os.path.join(video, image_filename.replace('.png', '.jpg')))
            img = Image.open(os.path.join('datasets/vipseg/VIPSeg_720P/imgs', vid_dict["file_names"][-1]))
            assert original_format.shape == np.array(img).shape[:2], f"Dismatch shape {original_format.shape} and {np.array(img).shape[:2]}"

            pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
            l = np.unique(original_format)

            # l: labels, el: entity labels
            for el in l:
                if el == 0:
                    continue
                if el < 125:
                    # stuff id
                    semantic_id = el
                    is_crowd = 0
                else:
                    # thing id
                    semantic_id = el // 100
                    is_crowd = 0

                semantic_id = semantic_id - 1
                if categories_dict[semantic_id]['isthing'] == 0:
                    is_crowd = 0
                mask = (original_format == el)

                if el not in instid2color:
                    segment_id, color = id_generator.get_id_and_color(semantic_id)
                    instid2color[el] = (segment_id, color)
                else:
                    segment_id, color = instid2color[el]

                pan_format[mask] = color
                if int(segment_id) not in annotations:
                    annotations[int(segment_id)] = {
                        "video_id": video_id,
                        "length": vid_len,
                        "id": int(segment_id),
                        "category_id": int(semantic_id) + 1,  # starts from 1
                        "bboxes": [None] * vid_len,
                        "areas": [None] * vid_len,
                        "iscrowd": is_crowd
                    }

                annotations[int(segment_id)]["areas"][i] = int(mask.sum())
                annotations[int(segment_id)]["bboxes"][i] = bounding_box(mask)

            if not os.path.exists(os.path.join(out_folder, video)):
                os.makedirs(os.path.join(out_folder, video))

            Image.fromarray(pan_format).save(os.path.join(out_folder, video, image_filename))

        print('image saved {}'.format(os.path.join(out_folder, video)))
        if video in v_train_list:
            v_train_videos.append(vid_dict)
            v_train_annotations += [v for k, v in annotations.items()]
        elif video in v_val_list:
            v_val_videos.append(vid_dict)
            v_val_annotations += [v for k, v in annotations.items()]
        elif video in v_test_list:
            v_test_videos.append(vid_dict)
            v_test_annotations += [v for k, v in annotations.items()]
        else:
            print(f"{video} not in train / val / test sets!!")

    for k, cate in categories_dict.items():
        cate["id"] = cate["id"] + 1
        categories_dict[k] = cate

    d_train = {
        'videos': v_train_videos,
        'annotations': v_train_annotations,
        'categories': categories,
    }
    save_json(d_train, out_file.replace(".json", "_train_cocovid.json"))
    print('==> Saved json file at %s' % (out_file.replace(".json", "_train_cocovid.json")))
    print(len(v_val_videos), len(v_val_list))
    assert len(v_val_videos) == len(v_val_list)
    d_val = {
        'videos': v_val_videos,
        'annotations': v_val_annotations,
        'categories': categories,
    }
    save_json(d_val, out_file.replace(".json", "_val_cocovid.json"))
    print('==> Saved json file at %s' % (out_file.replace(".json", "_val_cocovid.json")))

    d_test = {
        'videos': v_test_videos,
        'annotations': v_test_annotations,
        'categories': categories,
    }
    save_json(d_test, out_file.replace(".json", "_test_cocovid.json"))
    print('==> Saved json file at %s' % (out_file.replace(".json", "_test_cocovid.json")))


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # (x1, y1, w, h)


if __name__ == "__main__":
    panoptic_video_converter()
