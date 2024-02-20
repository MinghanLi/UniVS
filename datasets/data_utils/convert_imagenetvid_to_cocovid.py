import os
import json
import xmltodict

from videosam.data.datasets.imagenet_vid import ImageNetVID_CATEGORIES

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
imagenet_vid_path = os.path.join(_root, "ILSVRC2015/Data/VID/train/")
imagenet_vid_anno = os.path.join(_root, "ILSVRC2015/Annotations/VID/train/")
vid2vis_anno_json = os.path.join(_root, "ILSVRC2015/Annotations/VID/train.json")

# sampling frame every n
frame_interval = 20
if frame_interval > 1:
    vid2vis_anno_json = vid2vis_anno_json.replace('.json', '_every'+str(frame_interval)+'frame.json')

category_name2id = {cate['id_name']: cate['id'] for cate in ImageNetVID_CATEGORIES}
dataset = {
    'videos': [], 'annotations': [], 'categories': ImageNetVID_CATEGORIES,
    'info': {'description': 'ILSVRC_2015_VID',
             'url': 'https://image-net.org/challenges/LSVRC/2015/index.php',
             'version': '1.0', 'year': 2015},
    'licenses': [{'url': 'https://creativecommons.org/licenses/by/4.0/', 'id': 1,
                  'name': 'Creative Commons Attribution 4.0 License'}],
}

for root, dirs, files in os.walk(imagenet_vid_path):
    if len(files) == 0:
        continue

    root = '/'.join(root.split('/')[-2:])
    file_names = sorted(['/'.join([root, file]) for file in files])
    file_names = [file_names[t] for t in range(0, len(file_names), frame_interval)]

    # parse a xml file by name
    anno_file = imagenet_vid_anno+file_names[0].replace('.JPEG', '.xml')
    anno_data = open(anno_file, 'r').read()  # Read data
    img_size = xmltodict.parse(anno_data)['annotation']['size']  # Parse XML

    video = {'license': 1, 'coco_url': '', 'height': int(img_size['height']), 'width': int(img_size['width']),
             'length': len(file_names), 'file_names': file_names, 'flickr_url': '', 'id': root}
    dataset['videos'].append(video)

print("All videos are converted in ImageNet-VID.")

n_vid = 0
for root, dirs, files in os.walk(imagenet_vid_anno):
    if len(files) == 0:
        continue

    n_vid += 1
    if n_vid % 100 == 0:
        print('Processing the video: ', n_vid)

    root = '/'.join(root.split('/')[-2:])
    file_names = sorted(['/'.join([root, file]) for file in files])
    file_names = [file_names[t] for t in range(0, len(file_names), frame_interval)]

    anno_objs = {}
    for i, file_name in enumerate(file_names):
        anno_file = imagenet_vid_anno + file_name
        anno_data = open(anno_file, 'r').read()  # Read data
        annoDict = xmltodict.parse(anno_data)['annotation']
        if 'object' in annoDict:
            annoObjs = annoDict['object']  # Parse XML
            if isinstance(annoObjs, dict):
                annoObjs = [annoObjs]

            for annoObj in annoObjs:
                box = {k: float(v) for k, v in annoObj['bndbox'].items()}
                box_xywh = [0.5*(box['xmin'] + box['ymax']), 0.5*(box['ymin'] + box['ymax']),
                            box['xmax']-box['xmin'], box['ymax']-box['ymin']]
                box_area = box_xywh[2] * box_xywh[3]
                if annoObj['trackid'] not in anno_objs:
                    anno_objs[annoObj['trackid']] = {
                        'video_id': root, 'iscrowd': 0, 'occluded': int(annoObj['occluded']),
                        'category_id': category_name2id[annoObj['name']], 'id': int(annoObj['trackid']),
                        'height': int(annoDict['size']['height']), 'width': int(annoDict['size']['width']),
                        'length': len(file_names), 'bboxes': [[0.]*4]*i, 'areas': [0]*i}

                anno_objs[annoObj['trackid']]['bboxes'].append(box_xywh)
                anno_objs[annoObj['trackid']]['areas'].append(box_area)

        for anno_id, anno in anno_objs.items():
            if len(anno['bboxes']) < i+1:
                anno['bboxes'].append([0., 0., 0., 0.])
                anno['areas'].append([0.])

    for anno_id, anno in anno_objs.items():
        assert len(anno['bboxes']) == len(file_names), [len(anno['bboxes']), len(file_names)]
        dataset['annotations'].append(anno)

print("All annotations are converted in ImageNet-VID.")

print("Saving results to {}".format(vid2vis_anno_json))
with open(vid2vis_anno_json, 'w') as fp:
    json.dump(dataset, fp)

exit()


