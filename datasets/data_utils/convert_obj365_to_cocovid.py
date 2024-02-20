import json
import argparse
from os import walk


def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--src_json", default="datasets/objects365/annotations/zhiyuan_objv2_train.json", type=str,
                        help="")
    parser.add_argument("--des_json", default="datasets/objects365/annotations/zhiyuan_objv2_train_cocovid_250k.json",
                        type=str, help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Start loading annotations ...")
    src_dataset = json.load(open(args.src_json, 'r'))
    print("Finish loading annotations ... ")
    des_dataset = {
        'videos': [],
        'annotations': [],
        "categories": src_dataset["categories"]
    }

    inclusive_image_ids = []
    num_images = 0

    image_root = '/'.join(args.src_json.split('/')[:2] + ['images/train'])
    exit_filenames = next(walk(image_root), (None, None, []))[2]

    # convert image to videos
    num_imgs = len(src_dataset["images"])
    print(f'dataset has {num_imgs} images!')
    for i, img_dict in enumerate(src_dataset["images"]):
        if (i % int(num_imgs / 1000)) == 0:
            print(f'processed {i} of {num_imgs} images')

        if min(img_dict["width"], img_dict["height"]) < 480:
            continue

        file_name = img_dict["file_name"].split('/')[-1]
        if file_name not in exit_filenames:
            continue

        print("file names is ", img_dict["file_name"])
        des_dataset["videos"].append({
            "width": img_dict["width"],
            "height": img_dict["height"],
            "length": 1,
            "id": img_dict["id"],
            "file_names": [file_name]
        })

        inclusive_image_ids.append(img_dict["id"])
        num_images += 1

        # only use 250k images
        if len(inclusive_image_ids) > 250000:
            break

    assert len(inclusive_image_ids) > 0, 'Please check the number of selected images!'

    print("Finish convert images!")

    # convert annotations
    inclusive_image_ids_anno = []
    print("Start convert annotations!")
    num_annos = len(src_dataset["annotations"])
    for i, anno_dict in enumerate(src_dataset["annotations"]):
        if (i % int(num_imgs / 1000)) == 0:
            print(f'processed {i} of {num_annos} annotations')

        if anno_dict['image_id'] not in inclusive_image_ids:
            if len(inclusive_image_ids_anno) == len(inclusive_image_ids):
                break
            else:
                continue

        if anno_dict['image_id'] not in inclusive_image_ids_anno:
            inclusive_image_ids_anno.append(anno_dict['image_id'])

        des_dataset["annotations"].append({
            "iscrowd": anno_dict["iscrowd"],
            "category_id": anno_dict["category_id"],
            "id": anno_dict["id"],
            "video_id": anno_dict["image_id"],
            "bboxes": [anno_dict["bbox"]],  # [x, y, W, H], where (x, y) is the coordinates of left-top corner
            "areas": [anno_dict["area"]]
        })
    print("Finish convert annotations!")

    # save
    with open(args.des_json, "w") as f:
        json.dump(des_dataset, f)

    print(f'Done! Selected {len(des_dataset["videos"])} images over 480p')
