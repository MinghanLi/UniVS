def _get_vis_common_metadata():
    thing_ids = [k["id"] for k in MiXED_COMMON_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in MiXED_COMMON_CATEGORIES if k["isthing"] == 1]
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in MiXED_COMMON_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_vps_common_metadata():
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.

    stuff_classes = [k["name"] for k in MiXED_COMMON_CATEGORIES if k["isthing"] != 1]
    stuff_ids = [k["id"] for k in MiXED_COMMON_CATEGORIES if k["isthing"] != 1]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}

    all_colors = {k["id"]: k["color"] for k in MiXED_COMMON_CATEGORIES}
    stuff_colors = [all_colors[k["id"]] for k in MiXED_COMMON_CATEGORIES]

    thing_classes = [k["name"] for k in MiXED_COMMON_CATEGORIES if k["isthing"] == 1]
    thing_ids = [k["id"] for k in MiXED_COMMON_CATEGORIES if k["isthing"] == 1]
    thing_colors = [all_colors[k["id"]] for k in MiXED_COMMON_CATEGORIES if k["isthing"] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}

    meta = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
        "ignore_label": 255,
    }

    categories_ = {}
    for cat in MiXED_COMMON_CATEGORIES:
        categories_.update({cat['id']: cat})
    meta['categories'] = categories_

    return meta

def _get_vss_common_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.

    classes = [k["name"] for k in MiXED_COMMON_CATEGORIES]
    colors = [k["color"] for k in MiXED_COMMON_CATEGORIES]

    meta["stuff_classes"] = classes
    meta["stuff_colors"] = colors
    meta["thing_classes"] = None
    meta["thing_colors"] = None

    classes_id = [k['id'] for k in CATEGORIES]
    meta['stuff_classes_id'] = classes_id
    meta['thing_classes_id'] = None

    dataset_id_to_contiguous_id = {}
    for i, id_ in enumerate(classes_id):
        dataset_id_to_contiguous_id[id_] = i
    meta["stuff_dataset_id_to_contiguous_id"] = dataset_id_to_contiguous_id
    meta["thing_dataset_id_to_contiguous_id"] = None

    meta["ignore_label"] = 255

    return meta

MiXED_COMMON_CATEGORIES = [
    # person
    {'supercategory': 'person', "isthing": 1, "id": 0, "name": "person", "color": [220, 20, 60]},
    {'supercategory': 'person', "isthing": 1, "id": 1, "name": "hand", "color": [250, 170, 30]},
    {'supercategory': 'instrument',"isthing": 1, "id": 2, "name": "instrument", "color": [0, 255, 235]},
    # animal
    {'supercategory': 'animal', "isthing": 1, "id": 3, "name": "zebra", "color": [199, 100, 0]},
    {'supercategory': 'animal', "isthing": 1, "id": 4, "name": "tiger", "color": [166, 196, 102]},
    {'supercategory': 'animal', "isthing": 1, "id": 5, "name": "elephant", "color": [110, 76, 0]},
    {'supercategory': 'animal', "isthing": 1, "id": 6, "name": "Sheep", "color": [255, 208, 186]},
    {'supercategory': 'animal', "isthing": 1, "id": 7, "name": "squirrel", "color": [188, 208, 182]},
    {'supercategory': 'animal', "isthing": 1, "id": 8, "name": "frog", "color": [107, 142, 35]},
    {'supercategory': 'animal', "isthing": 1, "id": 9, "name": "lizard", "color": [119, 11, 32]},
    {'supercategory': 'animal', "isthing": 1, "id": 10, "name": "horse", "color": [182, 182, 255]},
    {'supercategory': 'animal', "isthing": 1, "id": 11, "name": "bear", "color": [174, 57, 255]},
    {'supercategory': 'animal', "isthing": 1, "id": 12, "name": "giraffe", "color": [72, 0, 118]},
    {'supercategory': 'animal', "isthing": 1, "id": 13, "name": "leopard", "color": [255, 179, 240]},
    {'supercategory': 'animal', "isthing": 1, "id": 14, "name": "fox", "color": [0, 125, 92]},
    {'supercategory': 'animal', "isthing": 1, "id": 15, "name": "deer", "color": [209, 0, 151]},
    {'supercategory': 'animal', "isthing": 1, "id": 16, "name": "owl", "color": [188, 218, 182]},
    {'supercategory': 'animal', "isthing": 1, "id": 17, "name": "turtle", "color": [0, 143, 149]},
    {'supercategory': 'animal', "isthing": 1, "id": 18, "name": "earless_seal", "color": [174, 255, 243]},
    {'supercategory': 'animal', "isthing": 1, "id": 19, "name": "fish", "color": [73, 77, 174]},
    {'supercategory': 'animal', "isthing": 1, "id": 20, "name": "whale", "color": [0, 228, 0]},
    {'supercategory': 'animal', "isthing": 1, "id": 21, "name": "shark", "color": [183, 130, 88]},
    {'supercategory': 'animal', "isthing": 1, "id": 22, "name": "rabbit", "color": [100, 170, 30]},
    {'supercategory': 'animal', "isthing": 1, "id": 23, "name": "duck", "color": [179, 0, 194]},
    {'supercategory': 'animal', "isthing": 1, "id": 24, "name": "cat", "color": [255, 77, 255]},
    {'supercategory': 'animal', "isthing": 1, "id": 25, "name": "cow, cattle", "color": [120, 166, 157]},  # merged
    {'supercategory': 'animal', "isthing": 1, "id": 26, "name": "monkey,ape", "color": [0, 60, 100]},
    {'supercategory': 'animal', "isthing": 1, "id": 27, "name": "dog", "color": [0, 226, 252]},
    {'supercategory': 'animal', "isthing": 1, "id": 28, "name": "snake", "color": [5, 121, 0]},
    {'supercategory': 'animal', "isthing": 1, "id": 29, "name": "giant_panda", "color": [0, 82, 0]},
    {'supercategory': 'animal', "isthing": 1, "id": 30, "name": "bat", "color": [0, 255, 41]},
    {'supercategory': 'animal', "isthing": 1, "id": 31, "name": "other_animal", "color": [0, 255, 133]},
    # bird
    {'supercategory': 'animal', "isthing": 1, "id": 32, "name": "bird", "color": [255, 109, 65]},
    {'supercategory': 'animal', "isthing": 1, "id": 33, "name": "eagle", "color": [0, 228, 5]},
    {'supercategory': 'animal', "isthing": 1, "id": 34, "name": "Poultry", "color": [117, 142, 35]},
    {'supercategory': 'animal', "isthing": 1, "id": 35, "name": "parrot", "color": [165, 42, 42]},
    # Transportation
    {'supercategory': 'Transportation', 'isthing': 1, 'id': 36, 'name': 'bus', "color": [250, 10, 15]},
    {'supercategory': 'Transportation', 'isthing': 1, 'id': 37, 'name': "truck", "color": [0, 0, 70]},
    {'supercategory': 'Transportation', 'isthing': 1, 'id': 38, 'name': "car", "color": [0, 0, 142]},
    {'supercategory': 'Transportation', "isthing": 1, "id": 39, "name": "motorbike, Motorcycle", "color": [0, 0, 230]},
    {'supercategory': 'Transportation', "isthing": 1, "id": 40, "name": "Bicycle", "color": [134, 134, 103]},
    {'supercategory': 'Transportation', 'isthing': 1, 'id': 41, 'name': 'train', "color": [0, 80, 100]},
    {'supercategory': 'Transportation', 'isthing': 1, 'id': 42, 'name': "boat, ship", "color": [0, 0, 192]},
    {'supercategory': 'Transportation', 'isthing': 1, 'id': 43, 'name': "raft", "color": [0, 0, 255]},
    {'supercategory': 'Transportation', "isthing": 1, "id": 44, "name": "airplane", "color": [106, 0, 228]},
    {'supercategory': 'Transportation', "isthing": 0, "id": 45, "name": "wheeled_machine", "color": [255, 224, 0]},
    # outdoor
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 46, 'name': 'traffic light', "color": [0, 173, 255]},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 47, 'name': 'roadblock', "color": [0, 163, 255]},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 48, 'name': 'fire hydrant', "color": [0, 153, 255]},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 49, 'name': 'stop sign', "color": [0, 143, 255]},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 50, 'name': 'parking meter', "color": [0, 133, 255]},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 51, 'name': 'bench', "color": [0, 123, 255]},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 52, 'name': 'windmill', "color": [0, 113, 255]},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 53, 'name': 'pole', "color": [0, 103, 255]},
    {'supercategory': 'outdoor', 'isthing': 1, 'id': 54, 'name': 'ladder', "color": [0, 93, 255]},
    # accessory
    {'supercategory': 'accessory', 'isthing': 1, 'id': 55, 'name': 'backpack', "color": [255, 0, 255]},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 56, 'name': 'umbrella, parasol', "color": [255, 20, 255]},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 57, 'name': 'handbag', "color": [255, 40, 255]},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 58, 'name': 'tie', "color": [255, 60, 255]},
    {'supercategory': 'accessory', 'isthing': 1, 'id': 59, 'name': 'suitcase, traveling_case, trolley_case', "color": [255, 80, 255]},  # merged
    # sports
    {'supercategory': 'sports', "isthing": 1, "id": 60, "name": "skateboard", "color": [134, 134, 103]},
    {'supercategory': 'sports', "isthing": 1, "id": 61, "name": "surfboard", "color": [145, 148, 174]},
    {'supercategory': 'sports', "isthing": 1, "id": 62, "name": "snowboard", "color": [133, 129, 255]},
    {'supercategory': 'sports', "isthing": 1, "id": 63, "name": "tennis_racket", "color": [5, 208, 186]},
    {'supercategory': 'sports', "isthing": 1, "id": 64, "name": "flying_disc, frisbee", "color": [250, 170, 32]},  # merged
    {'supercategory': 'sports', "isthing": 1, "id": 65, "name": "ball", "color": [0, 255, 173]},
    {'supercategory': 'sports', "isthing": 1, "id": 66, "name": "goal", "color": [40, 255, 173]},
    {'supercategory': 'sports', 'isthing': 1, 'id': 67, 'name': 'skis', "color": [80, 255, 173]},
    {'supercategory': 'sports', 'isthing': 1, 'id': 68, 'name': 'sports ball', "color": [120, 255, 173]},
    {'supercategory': 'sports', 'isthing': 1, 'id': 69, 'name': 'kite', "color": [160, 255, 173]},
    {'supercategory': 'sports', 'isthing': 1, 'id': 70, 'name': 'baseball bat', "color": [200, 255, 173]},
    {'supercategory': 'sports', 'isthing': 1, 'id': 71, 'name': 'baseball glove', "color": [240, 255, 173]},
    {'supercategory': 'sports', "isthing": 0, "id": 72, "name": "ball_net", "color": [0, 143, 255]},
    {'supercategory': 'sports', "isthing": 0, "id": 73, "name": "backboard", "color": [51, 255, 0]},
    # kitchen
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 74, 'name': 'bottle', "color": [255, 184, 184]},  # bottle_or_cup is splited as bottle and cup
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 75, 'name': 'wine glass', "color": [255, 164, 164]},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 76, 'name': 'cup', "color": [255, 144, 144]},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 77, 'name': 'fork', "color": [255, 124, 124]},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 78, 'name': 'knife', "color": [255, 104, 104]},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 79, 'name': 'spoon', "color": [255, 84, 84]},
    {'supercategory': 'kitchen', 'isthing': 1, 'id': 80, 'name': 'bowl', "color": [255, 64, 64]},
    {'supercategory': 'kitchen', "isthing": 1, "id": 81, "name": "plate", "color": [255, 44, 44]},
    {'supercategory': 'food', 'isthing': 1, 'id': 82, 'name': 'banana', "color": [255, 24, 24]},
    {'supercategory': 'food', 'isthing': 1, 'id': 83, 'name': 'apple', "color": [255, 4, 4]},
    {'supercategory': 'food', 'isthing': 1, 'id': 84, 'name': 'sandwich', "color": [125, 4, 4]},
    {'supercategory': 'food', 'isthing': 1, 'id': 85, 'name': 'orange', "color": [125, 24, 24]},
    {'supercategory': 'food', 'isthing': 1, 'id': 86, 'name': 'broccoli', "color": [125, 44, 44]},
    {'supercategory': 'food', 'isthing': 1, 'id': 87, 'name': 'carrot', "color": [125, 64, 64]},
    {'supercategory': 'food', 'isthing': 1, 'id': 88, 'name': 'hot dog', "color": [125, 84, 84]},
    {'supercategory': 'food', 'isthing': 1, 'id': 89, 'name': 'pizza', "color": [125, 104, 104]},
    {'supercategory': 'food', 'isthing': 1, 'id': 90, 'name': 'donut', "color": [125, 124, 124]},
    {'supercategory': 'food', 'isthing': 1, 'id': 91, 'name': 'cake', "color": [125, 144, 144]},
    {'supercategory': 'food-stuff', 'isthing': 0, 'id': 92, 'name': 'fruit', "color": [255, 204, 0]},
    {'supercategory': 'food-stuff', 'isthing': 0, 'id': 93, 'name': 'food-other-merged', "color": [255, 0, 143]},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 94, 'name': 'chair', "color": [0, 255, 82]},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 95, 'name': 'couch, sofa', "color": [0, 112, 255]},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 96, 'name': 'potted plant', "color": [0, 90, 255]},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 97, 'name': 'bed', "color": [0, 71, 255]},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 98, 'name': 'dining table', "color": [0, 49, 255]},
    {'supercategory': 'furniture', 'isthing': 1, 'id': 99, 'name': 'toilet, commode', "color": [0, 255, 163]},  # merged
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 100, 'name': 'bathtub', "color": [0, 194, 255]},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 101, 'name': 'counter', "color": [184, 24, 255]},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 102, 'name': 'door-stuff', "color": [184, 48, 255]},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 103, 'name': 'light', "color": [184, 72, 255]},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 104, 'name': 'mirror-stuff', "color": [184, 96, 255]},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 105, 'name': 'shelf', "color": [51, 0, 255]},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 106, 'name': 'stairs', "color": [51, 51, 255]},
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 107, 'name': 'cabinet-merged, cupboard, showcase, storage_rack', "color": [51, 102, 255]},  # merged
    {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 108, 'name': 'table-merged', "color": [0, 255, 194]},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 109, 'name': 'tv, screen, television', "color": [112, 224, 255]},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 110, 'name': 'laptop', "color": [112, 112, 255]},
    {'supercategory': 'electronic', "isthing": 1, "id": 111, "name": "computer", "color": [70, 184, 160]},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 112, 'name': 'mouse', "color": [35, 184, 160]},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 113, 'name': 'remote', "color": [70, 92, 160]},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 114, 'name': 'keyboard', "color": [71, 255, 0]},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 115, 'name': 'cell phone, Mobile_phone', "color": [153, 0, 255]},
    {'supercategory': 'electronic', 'isthing': 1, 'id': 116, 'name': 'other_electronic_product', "color": [255, 0, 163]},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 117, 'name': 'microwave', "color": [0, 163, 163]},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 118, 'name': 'oven', "color": [0, 82, 163]},  # merged
    {'supercategory': 'appliance', 'isthing': 1, 'id': 119, 'name': 'toaster, roaster', "color": [0, 163, 82]},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 120, 'name': 'sink', "color": [163, 163, 163]},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 121, 'name': 'refrigerator', "color": [255, 112, 0]},
    {'supercategory': 'appliance', 'isthing': 1, 'id': 122, 'name': 'washing_machine', "color": [143, 255, 0]},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 123, 'name': 'book', "color": [0, 214, 255]},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 124, 'name': 'clock', "color": [0, 184, 255]},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 125, 'name': 'vase, flower_pot', "color": [255, 0, 31]},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 126, 'name': 'scissors', "color": [126, 0, 31]},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 127, 'name': 'teddy bear', "color": [255, 31, 31]},
    {'supercategory': 'indoor', "isthing": 1, "id": 128, "name": "printer", "color": [163, 0, 255]},
    {'supercategory': 'indoor', "isthing": 1, "id": 129, "name": "box", "color": [173, 255, 0]},
    {'supercategory': 'indoor', "isthing": 1, "id": 130, "name": "basket", "color": [255, 92, 0]},
    {'supercategory': 'indoor', "isthing": 1, "id": 131, "name": "fan", "color": [163, 255, 0]},
    {'supercategory': 'indoor', "isthing": 1, "id": 132, "name": "fishbowl", "color": [0, 255, 61]},
    {'supercategory': 'indoor', "isthing": 1, "id": 133, "name": "painting_or_poster", "color": [0, 255, 92]},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 134, 'name': 'hair drier', "color": [0, 255, 46]},
    {'supercategory': 'indoor', 'isthing': 1, 'id': 135, 'name': 'toothbrush', "color": [0, 126, 92]},
    {'supercategory': 'indoor', "isthing": 0, "id": 136, "name": "toy", "color": [194, 255, 0]},
    {'supercategory': 'indoor', "isthing": 0, "id": 137, "name": "tissue", "color": [0, 224, 255]},
    {'supercategory': 'indoor', "isthing": 0, "id": 138, "name": "trash_can", "color": [255, 0, 245]},
    {'supercategory': 'indoor', "isthing": 0, "id": 139, "name": "lamp", "color": [31, 0, 255]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 140, 'name': 'banner', "color": [46, 255, 92]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 141, 'name': 'blanket', "color": [92, 255, 92]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 142, 'name': 'curtain', "color": [255, 235, 0]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 143, 'name': 'pillow', "color": [255, 0, 204]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 144, 'name': 'cushion_or_carpet', "color": [235, 12, 255]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 145, 'name': 'towel', "color": [235, 12, 126]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 146, 'name': 'rug-merged', "color": [167, 12, 255]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 147, 'name': 'clothes', "color": [133, 0, 255]},
    {'supercategory': 'textile', 'isthing': 0, 'id': 148, 'name': 'textiles', "color": [8, 184, 170]},
    {'supercategory': 'raw-material', 'isthing': 0, 'id': 149, 'name': 'cardboard', "color": [48, 184, 170]},
    {'supercategory': 'raw-material', 'isthing': 0, 'id': 150, 'name': 'paper-merged', "color": [88, 184, 170]},
    {'supercategory': 'structural', 'isthing': 0, 'id': 151, 'name': 'net', "color": [128, 184, 170]},
    {'supercategory': 'structural', 'isthing': 0, 'id': 152, 'name': 'fence-merged, handrail_or_fence', "color": [8, 184, 170]},
    {'supercategory': 'structural', 'isthing': 0, 'id': 153, 'name': 'pillar', "color": [235, 255, 7]},
    {'supercategory': 'plant', 'isthing': 0, 'id': 154, 'name': 'flower', "color": [255, 163, 0]},
    {'supercategory': 'plant', 'isthing': 0, 'id': 155, 'name': 'tree-merged', "color": [255, 143, 0]},
    {'supercategory': 'plant', 'isthing': 0, 'id': 156, 'name': 'grass-merged', "color": [255, 123, 0]},
    {'supercategory': 'plant', 'isthing': 0, 'id': 157, 'name': 'other_plant', "color": [255, 113, 0]},
    {'supercategory': 'building', 'isthing': 0, 'id': 158, 'name': 'bridge', "color": [255, 7, 71]},
    {'supercategory': 'building', "isthing": 0, "id": 159, "name": "tower", "color": [255, 27, 71]},
    {'supercategory': 'building', 'isthing': 0, 'id': 160, 'name': 'house', "color": [255, 47, 71]},
    {'supercategory': 'building', 'isthing': 0, 'id': 161, 'name': 'roof', "color": [255, 77, 71]},
    {'supercategory': 'building', 'isthing': 0, 'id': 162, 'name': 'tent', "color": [160, 150, 20]},
    {'supercategory': 'building', 'isthing': 0, 'id': 163, 'name': 'building-other-merged', "color": [255, 163, 0]},
    {'supercategory': 'ground', 'isthing': 0, 'id': 164, 'name': 'gravel', "color": [10, 255, 71]},
    {'supercategory': 'ground', 'isthing': 0, 'id': 165, 'name': 'platform', "color": [50, 255, 71]},
    {'supercategory': 'ground', 'isthing': 0, 'id': 166, 'name': 'playingfield, athletic_field', "color": [90, 255, 71]},  # merged
    {'supercategory': 'ground', 'isthing': 0, 'id': 167, 'name': 'railroad', "color": [130, 255, 71]},
    {'supercategory': 'ground', 'isthing': 0, 'id': 168, 'name': 'road, crosswalk', "color": [170, 255, 71]},
    {'supercategory': 'ground', 'isthing': 0, 'id': 169, 'name': 'sand', "color": [210, 255, 71]},
    {'supercategory': 'ground', 'isthing': 0, 'id': 170, 'name': 'snow, snowfield', "color": [250, 255, 71]},
    {'supercategory': 'ground', 'isthing': 0, 'id': 171, 'name': 'pavement-merged, path', "color": [10, 215, 71]},
    {'supercategory': 'ground', 'isthing': 0, 'id': 172, 'name': 'dirt-merged', "color": [10, 175, 71]},
    {'supercategory': 'ground', "isthing": 0, "id": 173, "name": "well_or_well_lid", "color": [9, 175, 230]},
    {'supercategory': 'water', 'isthing': 0, 'id': 174, 'name': 'river', "color": [224, 255, 8]},
    {'supercategory': 'water', "isthing": 0, "id": 175, "name": "lake", "color": [102, 8, 255]},
    {'supercategory': 'water', 'isthing': 0, 'id': 176, 'name': 'sea', "color": [7, 255, 255]},
    {'supercategory': 'water', 'isthing': 0, 'id': 177, 'name': 'ice', "color": [255, 184, 6]},
    {'supercategory': 'water', 'isthing': 0, 'id': 178, 'name': 'waterfall', "color": [255, 21, 6]},
    {'supercategory': 'water', 'isthing': 0, 'id': 179, 'name': 'water-other', "color": [255, 61, 76]},
    {'supercategory': 'sky', 'isthing': 0, 'id': 180, 'name': 'sky, sky-other-merged', "color": [155, 61, 6]},
    {'supercategory': 'window', 'isthing': 0, 'id': 181, 'name': 'window, window-other', "color": [255, 66, 36]},
    {'supercategory': 'ceiling', 'isthing': 0, 'id': 182, 'name': 'ceiling-merged', "color": [245, 31, 8]},
    {'supercategory': 'wall', 'isthing': 0, 'id': 183, 'name': 'wall, wall-other-merged', "color": [255, 91, 96]},  # merged
    {'supercategory': 'floor', 'isthing': 0, 'id': 184, 'name': 'floor-other-merged', "color": [155, 61, 45]},
    {'supercategory': 'wood', "isthing": 0, "id": 185, "name": "wood", "color": [7, 255, 224]},
    {'supercategory': 'solid', 'isthing': 0, 'id': 186, 'name': 'mountain-merged', "color": [55, 81, 6]},
    {'supercategory': 'solid', 'isthing': 0, 'id': 187, 'name': 'rock-merged', "color": [23, 61, 6]},
    {'supercategory': 'solid', 'isthing': 0, 'id': 188, 'name': 'stone', "color": [8, 255, 214]},
    # tool
    {"isthing": 1, "id": 189, "name": "flag", "color": [255, 5, 153]},
    {"isthing": 1, "id": 190, "name": "sculpture", "color": [0, 255, 20]},
    {"isthing": 1, "id": 191, "name": "barrel", "color": [0, 31, 255]},
    {"isthing": 1, "id": 192, "name": "gun", "color": [0, 122, 255]},
    {"isthing": 0, "id": 193, "name": "tyre", "color": [0, 235, 255]},
    {"isthing": 0, "id": 194, "name": "escalator", "color": [120, 120, 80]},
    {"isthing": 0, "id": 195, "name": "pipeline", "color": [255, 8, 41]},
    {"isthing": 0, "id": 196, "name": "Playground_slide", "color": [140, 140, 140]},
    {"isthing": 0, "id": 197, "name": "grandstand", "color": [255, 41, 10]},
    {"isthing": 0, "id": 198, "name": "billboard_or_Bulletin_Board", "color": [255, 122, 8]},
    {"isthing": 0, "id": 199, "name": "blackboard", "color": [92, 255, 0]},
    {"isthing": 0, "id": 200, "name": "cage", "color": [255, 0, 102]},
    {"isthing": 0, "id": 201, "name": "tool", "color": [255, 0, 112]},
]
