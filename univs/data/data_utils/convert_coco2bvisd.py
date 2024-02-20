import os
import json

from boxvis.data.datasets.bvisd import COCO_TO_BVISD

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

convert_list = [
    (
        COCO_TO_BVISD,
        os.path.join(_root, "coco/annotations/instances_train2017.json"),
        os.path.join(_root, "coco/annotations/coco2bvisd_train.json"),
        "COCO to BVISD:"
    ),
]

for convert_dict, src_path, out_path, msg in convert_list:
    src_f = open(src_path, "r")
    out_f = open(out_path, "w")
    src_json = json.load(src_f)
    # print(src_json.keys())   dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

    out_json = {}
    for k, v in src_json.items():
        if k != 'annotations':
            out_json[k] = v

    converted_item_num = 0
    out_json['annotations'] = []
    for anno in src_json['annotations']:
        if anno["category_id"] not in convert_dict:
            continue

        out_json['annotations'].append(anno)
        converted_item_num += 1

    json.dump(out_json, out_f)
    print(msg, converted_item_num, "items converted.")

