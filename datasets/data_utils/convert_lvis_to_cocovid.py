import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser("image to video converter")
    parser.add_argument("--src_json", default="datasets/lvis/lvis_v1_train.json", type=str, help="")
    parser.add_argument("--des_json", default="datasets/lvis/lvis_v1_train_video.json", type=str, help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_dataset = json.load(open(args.src_json, 'r')) 
    des_dataset = {'videos':[], 'categories':[], 'annotations':[]}
    des_dataset["categories"] = src_dataset["categories"]

    original_images = len(src_dataset["images"])

    included_images = []
    # videos with [h, w], where min(h, w) > 512, (remain 7089 images)
    for i, img_dict in enumerate(src_dataset["images"]):
        if (i % int(0.1*original_images)) == 0:
            print(f'processing {i*10} images')
        # if max(img_dict["width"], img_dict["height"]) < 360:
            # remove images with low resolution
            # continue

        included_images.append(int(img_dict["coco_url"].split('/')[-1][:-4]))

        vid_dict = {}
        vid_dict["length"] = 1
        vid_dict["file_names"] = ['/'.join(img_dict["coco_url"].split('/')[-2:])]
        vid_dict["width"], vid_dict["height"], vid_dict["id"] = img_dict["width"], img_dict["height"], img_dict["id"]
        vid_dict["neg_category_ids"] = img_dict["neg_category_ids"]
        des_dataset["videos"].append(vid_dict)

    print(len(included_images))

    # annotations
    len_anno = len(src_dataset["annotations"])
    for i, anno_dict in enumerate(src_dataset["annotations"]):
        if anno_dict['image_id'] not in included_images:
            continue
        if (i % int(0.1*len_anno)) == 0:
            print(f'processing {i*10} annotations')

        anno_dict_new = {}
        anno_dict_new["iscrowd"], anno_dict_new["category_id"], anno_dict_new["id"] = \
            0, anno_dict["category_id"], anno_dict["id"]
        anno_dict_new["video_id"] = anno_dict["image_id"]
        anno_dict_new["bboxes"] = [anno_dict["bbox"]]
        if "segmentation" in anno_dict:
            anno_dict_new["segmentations"] = [anno_dict["segmentation"]]
        anno_dict_new["areas"] = [anno_dict["area"]]
        des_dataset["annotations"].append(anno_dict_new)

    num_images = len(included_images)
    print(f'Select {num_images} images of {original_images}')
    print(args.des_json)
    # save
    with open(args.des_json, "w") as f:
        json.dump(des_dataset, f)
    print("Done!")
    exit()