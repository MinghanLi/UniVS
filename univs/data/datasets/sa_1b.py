import json
import logging
import numpy as np
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)


def load_sa_1b_json(anno_file_json, image_root):
    """
    Args:
        anno_file_json: all names of annotation files
        image_root: the root of SA-1B dataset

    Returns:
        dataset_dicts: same format as COCO

    """

    # Load annotations of per image in dataset_mapper.py, cause SA-1B is too large!
    anno_files = json.load(open(anno_file_json, 'r'))

    # anno_files["annotation_names"]: List[anno_file, anno_file, ....]
    # {
    #     "image": image_info,
    #     "annotations": [annotation],
    # }
    #
    # image_info
    # {
    #     "image_id": int,  # Image id
    #     "width": int,  # Image width
    #     "height": int,  # Image height
    #     "file_name": str,  # Image filename
    # }
    #
    # annotation
    # {
    #     "id": int,  # Annotation id
    #     "segmentation": dict,  # Mask saved in COCO RLE format.
    #     "bbox": [x, y, w, h],  # The box around the mask, in XYWH format
    #     "area": int,  # The area in pixels of the mask
    #     "predicted_iou": float,  # The model's own prediction of the mask's quality
    #     "stability_score": float,  # A measure of the mask's quality
    #     "crop_box": [x, y, w, h],  # The crop of the image used to generate the mask, in XYWH format
    #     "point_coords": [[x, y]],  # The point coordinates input to the model to generate the mask
    # }

    return anno_files["annotation_names"]


def register_sa_1b_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_sa_1b_json(json_file, image_root))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


if __name__ == "__main__":
    """
    Test the SA-1B json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from PIL import Image

    logger = setup_logger(name=__name__)
    # assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ytvis_2019_train")

    anno_file_json = "/data0/sa_1b/SA-1B-Downloader/annotations_5k_1.json"
    image_root = "/data0/sa_1b/SA-1B-Downloader"
    dicts = load_sa_1b_json(anno_file_json, image_root)
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "/data0/sa_1b/SA-1B-Downloader/visual/"
    os.makedirs(dirname, exist_ok=True)

    for d in dicts:
        file_name = d["file_name"]
        img = np.array(Image.open('/'.join([image_root, 'images', file_name])))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, file_name.split('/')[-1])
        vis.save(fpath)
