import os
import json
import time

from ..datasets.bvisd import BVISD_CATEGORIES, YTVIS_2021_TO_BVISD, OVIS_TO_BVISD


_root = os.getenv("DETECTRON2_DATASETS", "datasets")

annotation_files = [
    (
        'ytvis_2021',
        os.path.join(_root, "ytvis_2021/train/JPEGImages"),
        os.path.join(_root, "ytvis_2021/valid_sub.json"),
        YTVIS_2021_TO_BVISD
    ),
    (
        'ovis',
        os.path.join(_root, "ovis/train/JPEGImages"),
        os.path.join(_root, "ovis/valid_sub.json"),
        OVIS_TO_BVISD
    )
]

dataset_bvisd = {
    'info': {'description': 'YTVIS21 + OVIS + COCO with box annotations',
             'url': 'https://youtube-vos.org/challenge/2021/, '
                    'https://ovis-workshop.github.io/,',
             'version': '1.0', 'year': 2023},
    "categories": BVISD_CATEGORIES, 
    "videos": [], 
    "annotations": []}

num_saved_videos = 0
for data_name, root_path, anno_file, category_map_dict in annotation_files:
    print(f'Loading {data_name} annotations into memory...')
    tic = time.time()
    dataset = json.load(open(anno_file, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Finish loading (t={:0.2f}s)'.format(time.time() - tic))

    # map video ids from source dataset to bvisd dataset
    video_ids = sorted([video_dict["id"] for video_dict in dataset["videos"]])
    video_ids_mapped = range(num_saved_videos, num_saved_videos + len(video_ids))
    video_ids_map_dict = {vid: vid_mapped for vid, vid_mapped in zip(video_ids, video_ids_mapped)}

    for video_dict in dataset["videos"]:
        # add file paths into the file names
        video_dict["file_names"] = [os.path.join(root_path, file_path) for file_path in video_dict["file_names"]]
        video_dict["id"] = video_ids_map_dict[video_dict["id"]]

    for anno in dataset['annotations']:
        anno['category_id'] = category_map_dict[anno['category_id']]
        anno['video_id'] += video_ids_map_dict[anno['video_id']]

    dataset_bvisd["videos"] += dataset["videos"]
    dataset_bvisd["annotations"] += dataset["annotations"]

    print(f'Done for {data_name}')

os.makedirs(os.path.join(_root, "bvisd"), exist_ok=True)
bvisd_valid_anno_json = os.path.join(_root, "bvisd", "valid_sub.json")
print("Saving annotations to {}".format(bvisd_valid_anno_json))
with open(bvisd_valid_anno_json, 'w') as fp:
    json.dump(dataset_bvisd, fp)

print('Finish all!')

    