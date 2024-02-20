import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--src_json", default="datasets/refcoco/refcoco-mixed/instances_train.json", type=str, help="")
    parser.add_argument("--des_json", default="datasets/refcoco/refcoco-mixed/instances_train_video360p.json", type=str,
                        help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_dataset = json.load(open(args.src_json, 'r'))
    des_dataset = {
        'videos': [],
        'annotations': [],
        "categories": src_dataset["categories"]
    }

    num_imgs = len(src_dataset["images"])

    # videos
    inclusive_image_ids = []
    for i, img_dict in enumerate(src_dataset["images"]):
        if (i % int(num_imgs / 10)) == 0:
            print(f'processed {i} of {num_imgs} images')

        if max(img_dict["width"], img_dict["height"]) < 360:
            # pass images with low resolution
            continue

        img_id = img_dict["id"]
        if img_id not in inclusive_image_ids:
            inclusive_image_ids.append(img_id)

            vid_dict = {
                "length": 1,
                "file_names": [img_dict["file_name"].split('_')[-1]],
                "width": img_dict["width"],
                "height": img_dict["height"],
                "id": img_dict["id"]
            }
            des_dataset["videos"].append(vid_dict)

    num_annos = len(src_dataset["annotations"])
    for i, anno_dict in enumerate(src_dataset["annotations"]):
        if (i % int(num_annos / 10)) == 0:
            print(f'processed {i} of {num_annos} annotations')

        img_id = anno_dict["image_id"]
        if img_id in inclusive_image_ids:
            anno_dict_new = {
                "iscrowd": anno_dict["iscrowd"],
                "category_id": anno_dict["category_id"],
                "id": anno_dict["id"],
                "video_id": img_id,
                "bboxes": [anno_dict["bbox"]],
                "expressions": [anno_dict["expression"]],
                "exp_id": anno_dict["exp_id"]
            }
            if "segmentation" in anno_dict:
                anno_dict_new["segmentations"] = [anno_dict["segmentation"]]
            anno_dict_new["areas"] = [anno_dict["area"]]

            des_dataset["annotations"].append(anno_dict_new)

    # save
    print("save:", args.des_json)
    with open(args.des_json, "w") as f:
        json.dump(des_dataset, f)

    # Done! Selected 55809 images and 321007 expressions
    print(f'Done! Selected {len(des_dataset["videos"])} images and {len(des_dataset["annotations"])} expressions')
