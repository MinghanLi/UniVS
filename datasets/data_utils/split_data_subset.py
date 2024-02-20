import json
import time
import random
import argparse

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("split datasets")
    parser.add_argument("--annotation_file", default="datasets/lvis/lvis_v1_val.json", type=str)
    parser.add_argument("--out_file", default="datasets/lvis/lvis_v1_val_1000.json", type=str)
    parser.add_argument("--split_precent", default=0.05, type=float)
    parser.add_argument("--task", default='split_valid_sub_from_train_image', type=str)
    return parser.parse_args()


def split_valid_sub_from_train_image(annotation_file, out_file, split_precent=0.1):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(annotation_file, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    n_images_valid_sub = int(split_precent * len(dataset['images']))

    dataset_valid_sub = {'images': dataset['images'][:n_images_valid_sub], 'annotations': []}
    for k, v in dataset.items():
        if k not in dataset_valid_sub:
            dataset_valid_sub[k] = v

    n_images = len(dataset_valid_sub['images'])
    print(f'Spliting {n_images} images as valid_sub set')

    image_ids_valid_sub = [
        int(im_dict['coco_url'].split('/')[-1].replace('.jpg', ''))
        for im_dict in dataset_valid_sub['images']
    ]

    for anno in dataset['annotations']:
        if anno['image_id'] in image_ids_valid_sub:
            dataset_valid_sub['annotations'].append(anno)

    print("Saving results to {}".format(out_file))
    with open(out_file, 'w') as fp:
        json.dump(dataset_valid_sub, fp)


def downsampling_train_set_video(annotation_file, precent=0.1):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(annotation_file, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    videos = dataset['videos']
    annotations = dataset['annotations']

    dataset_ds = {'videos': [], 'annotations': []}
    for k, v in dataset.items():
        if k not in dataset_ds:
            dataset_ds[k] = v

    for video in videos:
        video_id = video['id']
        length = len(video['file_names'])

        length_ds = max(int(length * precent), 1)

        ds_type = 'uniform'
        if ds_type == 'random':
            # randomly sampling frames
            sample_idx = random.sample(range(length), length_ds)
            sample_idx.sort()
        else:
            stride = length // length_ds
            sample_idx = range(stride//2, length, stride)
        vid_ds = {'length': length_ds, 'file_names': [video['file_names'][idx] for idx in sample_idx]}

        for k, v in video.items():
            if k not in vid_ds:
                vid_ds[k] = v

        dataset_ds['videos'].append(vid_ds)
        for anno in annotations:
            if anno['video_id'] == video_id:
                segm = [anno['segmentations'][idx] for idx in sample_idx]
                if segm != [None] * length_ds:
                    anno_ds = {k: [v[idx] for idx in sample_idx] if isinstance(v, list) else v for k, v in anno.items()}
                    assert len(sample_idx) == len(anno_ds['segmentations'])
                    dataset_ds['annotations'].append(anno_ds)

    file_path = annotation_file[:-5] + '_downsample' + str(int(precent*100)) + '%.json'
    print("Saving results to {}".format(file_path))
    with open(file_path, 'w') as fp:
        json.dump(dataset_ds, fp)


def split_valid_sub_from_train_video(annotation_file, split_precent=0.1):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(annotation_file, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    n_videos_cate_valid_sub = [0] * len(dataset['categories'])

    videos = dataset['videos']
    annotations = dataset['annotations']
    n_videos_each_cate_valid_sub = 2 * int(len(videos) * split_precent) // len(dataset['categories'])
    if annotation_file.split('/')[-2] == 'ovis':
        n_videos_each_cate_valid_sub *= 2

    dataset_train_sub = {'videos': [], 'annotations': []}
    for k, v in dataset.items():
        if k not in dataset_train_sub:
            dataset_train_sub[k] = v

    dataset_valid_sub = {'videos': [], 'annotations': []}
    for k, v in dataset.items():
        if k not in dataset_valid_sub:
            dataset_valid_sub[k] = v

    video_ids_train_sub, video_ids_valid_sub = [], []
    for anno in annotations:
        if anno['video_id'] not in video_ids_train_sub+video_ids_valid_sub:
            if n_videos_cate_valid_sub[anno['category_id']-1] <= n_videos_each_cate_valid_sub and len(anno["segmentations"]) <= 100:
                video_ids_valid_sub.append(anno['video_id'])
            else:
                video_ids_train_sub.append(anno['video_id'])

        if anno['video_id'] in video_ids_valid_sub:
            dataset_valid_sub['annotations'].append(anno)
            n_videos_cate_valid_sub[anno['category_id'] - 1] += 1
        else:
            dataset_train_sub['annotations'].append(anno)

    for video in videos:
        if video['id'] in video_ids_valid_sub:
            dataset_valid_sub['videos'].append(video)
        else:
            dataset_train_sub['videos'].append(video)

    file_path_train_sub = annotation_file.replace('train.json', 'train_sub.json')
    file_path_valid_sub = annotation_file.replace('train.json', 'valid_sub.json')
    print(len(dataset['annotations']), len(dataset_train_sub['annotations']), len(dataset_valid_sub['annotations']))
    print("Saving results to {}".format(file_path_train_sub))
    with open(file_path_train_sub, 'w') as fp:
        json.dump(dataset_train_sub, fp)
    print("Saving results to {}".format(file_path_valid_sub))
    with open(file_path_valid_sub, 'w') as fp:
        json.dump(dataset_valid_sub, fp)


def instance_annotation_analysis(annotation_file):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(annotation_file, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    if 'videos' in dataset:
        n_frames = sum([v['length'] for v in dataset['videos']])
        print('Total number of frames:', n_frames)

    img_ids = []
    num_classes = len(dataset['categories'])
    num_insts_each_classes = [0] * num_classes
    annotations = dataset['annotations']
    for anno in annotations:
        if anno['category_id'] > len(num_insts_each_classes):
            num_insts_each_classes += [0] * (anno['category_id']-len(num_insts_each_classes))
        if "segmentations" in anno:
            for segm in anno["segmentations"]:
                if segm is not None:
                    num_insts_each_classes[anno['category_id'] - 1] += 1
        elif "bboxes" in anno:
            for box in anno["bboxes"]:
                if sum(box) > 0:
                    num_insts_each_classes[anno['category_id'] - 1] += 1
        elif "bbox" in anno:
            if sum(anno["bbox"]) > 0:
                num_insts_each_classes[anno['category_id'] - 1] += 1
                img_ids.append(anno['image_id'])

    print('Total number images in Coco2JVIS:', len(set(img_ids)))

    barWidth = 0.5
    plt.bar(range(1, len(num_insts_each_classes)+1), num_insts_each_classes, color='grey', width=barWidth, label='num_inst')
    # plt.show()

    plt.savefig(annotation_file.replace('.json', '_num_inst_each_class.jpg'))
    plt.clf()

    print(num_insts_each_classes, sum(num_insts_each_classes))
    with open(annotation_file.replace('.json', '_num_inst_each_class.txt'), 'w') as fp:
        fp.write(str(num_insts_each_classes))


if __name__ == "__main__":
    # annotation_file = "datasets/coco/annotations/instances_val2017.json"
    # out_file = "datasets/coco/annotations/instances_val2017_500.json"
    args = parse_args()

    if args.task == 'downsample_train_video':
        for precent in [0.01, 0.1]:
            downsampling_train_set_video(args.annotation_file, precent)

    elif args.task == 'split_valid_sub_from_train_video':
        split_valid_sub_from_train_video(args.annotation_file, args.split_precent)

    elif args.task == 'instance_annotation_analysis':
        instance_annotation_analysis(args.annotation_file)

    elif args.task == 'split_valid_sub_from_train_image':
        split_valid_sub_from_train_image(args.annotation_file, args.out_file, args.split_precent)

    else:
        NotImplementedError()

    exit()