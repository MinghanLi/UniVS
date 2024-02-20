import os, sys
import json
import glob
import numpy as np
import PIL.Image as Image
from tqdm import trange
from panopticapi.utils import IdGenerator, save_json
import argparse

ROOT_DIR = 'datasets/VSPW_480p/'
with open('datasets/VSPW_480p/label_num_dic_final.json', 'r') as f:
    CATEGORIES = json.load(f)


def semantic_video_converter():
    data_folder = os.path.join(ROOT_DIR, 'data')

    with open(os.path.join(ROOT_DIR, 'train.txt'), 'r') as f:
        lines = f.readlines()
        v_train_list = [line[:-1] for line in lines]
    v_train_videos = []
    v_train_annotations = []

    with open(os.path.join(ROOT_DIR, 'val.txt'), 'r') as f:
        lines = f.readlines()
        v_val_list = [line[:-1] for line in lines]
    v_val_videos = []
    v_val_annotations = []

    with open(os.path.join(ROOT_DIR, 'test.txt'), 'r') as f:
        lines = f.readlines()
        v_test_list = [line[:-1] for line in lines]
    v_test_videos = []
    v_test_annotations = []
    
    print(CATEGORIES)
    # 0 -> others
    categories = [{'id': int(c_id), 'name': c_name} for c_name, c_id in CATEGORIES.items() if int(c_id) > 0]

    video_id = -1
    for video_name in sorted(os.listdir(data_folder)):
        if video_name not in v_val_list:
            continue

        print('processing video:{}'.format(video_name))
        video_id += 1

        image_dir = os.path.join(data_folder, video_name, 'origin')
        mask_dir = os.path.join(data_folder, video_name, 'mask')
        image_filenames = sorted(os.listdir(image_dir))
        mask_filenames = sorted(os.listdir(mask_dir))
        assert len(image_filenames) == len(mask_filenames), 'Mismatch length of file names between image and mask!'

        vid_len = len(image_filenames)
        origin_image = np.array(
            Image.open(os.path.join(image_dir, image_filenames[0]))
        )
        vid_dict = {
            # "id": video_id,
            "id": video_name,  # need video name for evaluation
            "length": vid_len,
            "width": origin_image.shape[1],
            "height": origin_image.shape[0],
            "file_names": [os.path.join(video_name, 'origin', image_filename) for image_filename in image_filenames]
        }
        v_val_videos.append(vid_dict)

    d_val = {
        'videos': v_val_videos,
        'annotations': None,
        'categories': categories,
    }
    save_json(d_val, os.path.join(ROOT_DIR, "val_cocovid.json"))
    print('==> Saved json file at %s' % (os.path.join(ROOT_DIR, "val_cocovid.json")), len(v_val_videos))


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]  # (x1, y1, w, h)


if __name__ == "__main__":
    semantic_video_converter()
