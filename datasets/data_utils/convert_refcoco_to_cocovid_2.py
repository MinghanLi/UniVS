import os
import json

if __name__ == "__main__":
    merged_dir = "datasets/refcoco/refcoco-mixed"
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    merged_json = "datasets/refcoco/refcoco-mixed/instances_train.json"
    new_data = {"images": [], "annotations": []}

    exp_idx = 0  # index of the instance
    image_id = 0
    for dataset in ["refcoco", "refcocog", "refcoco+"]:
        print(f"processing {dataset} ...")
        json_path = "datasets/refcoco/%s/instances_%s_train.json" % (dataset,dataset)
        data = json.load(open(json_path, 'r'))

        new_data["categories"] = data["categories"]

        image_id_map = {}
        for img in data["images"]:
            image_id_map[img["id"]] = image_id
            img["id"] = image_id
            image_id += 1
            new_data["images"].append(img)

        # for split in data.keys():
        for anno in data["annotations"]:
            exp_idx = exp_idx + 1
            anno["exp_id"] = exp_idx
            anno["image_id"] = image_id_map[anno["image_id"]]
            new_data["annotations"].append(anno)

        print(f"Finish {dataset}.")

    print("Done {} images and {} expressions".format(len(new_data["images"]), len(new_data["annotations"])))

    json.dump(new_data, open(merged_json, 'w'))  # 321327 referred objects