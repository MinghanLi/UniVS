import json
import os
import cv2
import torch
import numpy as np

from PIL import Image
from detectron2.structures import PolygonMasks
import pycocotools.mask as maskUtils

def converter():
    _root = '/home/notebook/data/personal/S9053736/code/UniVS/datasets/vipseg/VIPSeg_720P/'
    json_file = 'panoptic_gt_VIPSeg_val.json'
    txt_file = 'val_sub.txt'

    with open(os.path.join(_root, txt_file), 'r') as file:
        video_names = file.readlines()
    video_names_sub = [vn.replace('\n', '') for vn in video_names]

    data_dict = json.load(open(os.path.join(_root, json_file), 'r'))
    videos = data_dict['videos']
    annos = data_dict['annotations']
    categories = data_dict['categories']
    print(len(videos), len(annos))
    print(videos[0])

    if 'file_names' in videos[0]:  # cocovid
        id_name_map = {vid['id']: vid['file_names'][0].split('/')[0] for vid in videos}  
        print(len(id_name_map))
        
    videos_sub = []
    annos_sub = []
    for vid in videos:
        if 'file_names' in vid:  # cocovid
            vn = vid['file_names'][0].split('/')[0]
        else:
            vn = vid['video_id']
        if vn in video_names_sub:
            videos_sub.append(vid)
    print(len(videos_sub))

    for anno in annos:
        vid_id = anno['video_id']
        if 'file_names' in videos[0]:  # cocovid
            vid_id = id_name_map[vid_id]
        if vid_id in video_names_sub:
            annos_sub.append(anno)
    print(len(annos_sub))

    sub_data_dict = {'videos': videos_sub, 'annotations': annos_sub, 'categories': data_dict['categories']}

    save_path = os.path.join(_root, json_file.replace('val', txt_file[:-4]))
    print('Saving:', save_path)

    with open(save_path, 'w') as file:
        json.dump(sub_data_dict, file)


if __name__ == "__main__":
    converter()