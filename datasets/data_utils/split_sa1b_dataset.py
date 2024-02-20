import os.path
from os import walk
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser("Split SA1B into 100k subsets for dataloader.")
    parser.add_argument("--data_path", default='datasets/sa_1b/', type=str, help="path of SA-1B dataset")
    parser.add_argument("--num_frames_per_file", default=100000, type=int, help="the number of images in per file")
    parser.add_argument("--num_files_merged", default=5, type=int, help="the number of images in per file")
    parser.add_argument("--task_type", default='split', type=str, help="")
    return parser.parse_args()


def merge_sa1b_dataset(args):
    num_files_merged = args.num_files_merged
    num_k = int(args.num_frames_per_file / 1000)
    num_k_merged = num_k*num_files_merged

    anno_root = os.path.join(args.data_path, 'annotations_{}k'.format(str(num_k)))
    save_dir = os.path.join(args.data_path, 'annotations_{}k'.format(str(num_k_merged)))
    os.makedirs(save_dir, exist_ok=True)

    n = 0
    filenames = next(walk(anno_root), (None, None, []))[2]
    dataset = {'annotation_names': []}
    for file in filenames:
        n += 1
        anno = json.load(open(os.path.join(anno_root, file), 'r'))
        dataset['annotation_names'] += anno['annotation_names']

        if n % num_files_merged == 0:
            save_file_name = 'annotations_{}k'.format(str(num_k_merged)) + '_' + str(n // num_files_merged) + '.json'
            print(os.path.join(save_dir, save_file_name))
            assert len(dataset['annotation_names']) == num_k_merged * 1000
            with open(os.path.join(save_dir, save_file_name), "w") as f:
                json.dump(dataset, f)

            dataset = {'annotation_names': []}


def split_sa1b_dataset(args):
    _root = args.data_path

    im_path = os.path.join(_root, 'images')
    im_filenames = next(walk(im_path), (None, None, []))[2]

    anno_path = os.path.join(_root, 'annotations')
    anno_filenames = next(walk(anno_path), (None, None, []))[2]

    dataset = {'annotation_names': []}

    n = 0
    n_frames_per_file = args.num_frames_per_file
    for i, im_file in enumerate(im_filenames):
        if im_file.replace('.jpg', '.json') not in anno_filenames:
            continue

        n += 1
        dataset['annotation_names'].append(im_file.replace('.jpg', '.json'))

        if n % n_frames_per_file == 0 or i == len(im_filenames) - 1:
            save_path = anno_path + '_' + str(n_frames_per_file//1000) +'k_'+ str(n//n_frames_per_file) +'.json'
            print(save_path)
            with open(save_path, "w") as f:
                json.dump(dataset, f)

            dataset = {'annotation_names': []}

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    if args.task_type == 'split':
        split_sa1b_dataset(args)
    else:
        merge_sa1b_dataset(args)