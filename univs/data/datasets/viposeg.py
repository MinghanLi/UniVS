from .vps import VIPseg_CATEGORIES 


def _get_viposeg_metadata_val():
    """
    Args:
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
    """
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.

    stuff_classes = [k["name"] for k in VIPseg_CATEGORIES if k["isthing"] != 1]
    stuff_ids = [k["id"] for k in VIPseg_CATEGORIES if k["isthing"] != 1]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}

    all_colors = {k["id"]: k["color"] for k in VIPseg_CATEGORIES}
    stuff_colors = [all_colors[k["id"]] for k in VIPseg_CATEGORIES]

    thing_classes = [k["name"] for k in VIPseg_CATEGORIES if k["isthing"] == 1]
    thing_ids = [k["id"] for k in VIPseg_CATEGORIES if k["isthing"] == 1]
    thing_colors = [all_colors[k["id"]] for k in VIPseg_CATEGORIES if k["isthing"] == 1]
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
    for cat in VIPseg_CATEGORIES:
        categories_.update({cat['id']: cat})
    meta['categories'] = categories_

    return meta