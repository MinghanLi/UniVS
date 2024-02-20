import json
import time


def split_valid_sub_from_train(annotation_file, valid_precent=0.1):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(annotation_file, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    n_videos_cate_valid_sub = [0] * len(dataset['categories'])

    videos = dataset['videos']
    annotations = dataset['annotations']
    n_videos_each_cate_valid_sub = 2 * int(len(videos) * valid_precent) // len(dataset['categories'])
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


if __name__ == "__main__":
    annotation_file_list = [
        "/data/VIS/BoxVIS/datasets/ytvis_2021/train.json",
        "/data/VIS/BoxVIS/datasets/ovis/train.json",
    ]

    for annotation_file in annotation_file_list:
        split_valid_sub_from_train(annotation_file, valid_precent=0.1)

    exit()