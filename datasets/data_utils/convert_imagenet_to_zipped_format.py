import argparse
import os

from imagenet_label_to_wordnet_synset import wordnet_to_class_id


def parse_args():
    parser = argparse.ArgumentParser("convert ImageNet to zipped format to boost the slow speed "
                                     "when reading images from massive small files")
    parser.add_argument("--src_dir", default="datasets/ImageNet/", type=str, help="")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    id_map = wordnet_to_class_id()

    data_types = ['train', 'val']
    for data_type in data_types:
        out_file = os.path.join(args.src_dir, data_type+'_map.txt')
        data_dir = os.path.join(args.src_dir, data_type)

        with open(out_file, 'w') as f:
            for root, dirs, files in os.walk(data_dir):
                for class_dir in dirs:
                    if not class_dir[1:] + '-n' in id_map:
                        continue

                    label = id_map[class_dir[1:] + '-n']
                    data_class_dir = os.path.join(data_dir, class_dir)
                    print(f'class {class_dir} with id {label} has {len(os.listdir(data_class_dir))} images')
                    for img_name in os.listdir(data_class_dir):
                        input_info = os.path.join(class_dir, img_name) + f'  {label}\n'
                        print(input_info[:2])
                        f.write(input_info)

