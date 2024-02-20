import glob, os
import json
import pycocotools.mask as mask_util

from videosam.evaluation import YTVISEvaluator


def convert_gt_annotations_class_agnostic(dataset_root_dir=None):
    if dataset_root_dir is None:
        dataset_root_dir = 'datasets/ytvis_2021'

    os.chdir(dataset_root_dir)
    for file in glob.glob("valid_sub_*"):
        # load mata file for dataset
        if '_class_agnostic.json' not in file and file[-5:] == '.json':
            dataset = json.load(open(file, 'r'))

            dataset['categories'] = dataset['categories'][:1]
            dataset['categories'][0]['name'] = 'all'
            for ann in dataset['annotations']:
                ann['category_id'] = 1

            output_file = file.replace('.json', '_class_agnostic.json')
            print('save class_agnostic valid subset in:', output_file)
            with open(output_file, 'w') as fp:
                json.dump(dataset, fp)

            del dataset

# Step 1: predict masks by SAM, saved in results.json


def convert_predictions_to_pseudo_anno(dataset_meta_file=None, prediction_file=None):
    """
    Step 2: convert predicted masks as pseudo annotations to get categories by RegionCLIP,
            the file is saved as pseudo_anno_regionclip.json
    Returns:

    """
    # load mata file for dataset
    if dataset_meta_file is None:
        dataset_meta_file = 'datasets/ytvis_2021/valid_sub_0.01.json'
    dataset = json.load(open(dataset_meta_file, 'r'))

    # load predicted objects with masks
    if prediction_file is None:
        prediction_file = 'output/video_sam/inference/results.json'
    predictions = json.load(open(prediction_file, 'r'))
    for id, pred in enumerate(predictions):
        pred['id'] = id + 1
        pred['iscrowd'] = 0

        if 'bboxes' not in pred:
            # Get bounding boxes surrounding encoded masks.
            pred['bboxes'] = mask_util.toBbox(pred['segmentations']).tolist()  # xywh, (x, y) is the left-top point
            pred['areas'] = mask_util.area(pred['segmentations']).tolist()

        if 'height' not in pred:
            for vid in dataset["videos"]:
                if vid['id'] == pred['video_id']:
                    pred['height'] = vid['height']
                    pred['width'] = vid['width']

    # predicted masks as pseudo annotation, used to get categories by RegionCLIP
    dataset['annotations'] = predictions
    output_file = prediction_file.replace('results.json', 'pseudo_anno_regionclip.json')
    with open(output_file, 'w') as fp:
        json.dump(dataset, fp)

    print(f'Save predicted masks as pseudo annotation in {output_file}')


def evaluate_predictions_on_ytvis(dataset_name, pred_file, output_folder='output/inference/'):
    """
    # Step 3: if the dataset has ground-truth annotation ==> metric performance,
    # otherwise submit the annotations in pseudo_anno_regionclip.json to evaluate results
    Returns:
    """

    Eval = YTVISEvaluator(dataset_name, output_dir=output_folder)
    Eval.eval_predictions_by_files(pred_file)


if __name__ == '__main__':
    step = 3

    if step == 1:
        convert_gt_annotations_class_agnostic()

    if step == 2:
        convert_predictions_to_pseudo_anno()

        # Move to the path of RegionCLIP
        # RegionCLIP$ cp ../BoxVIS/output/video_sam/inference/pseudo_anno_regionclip.json output/vis/
        # RegionCLIP$ sh test_zeroshot_inference.sh
        # RegionCLIP$ cp output/inference/results.json ../BoxVIS/output/video_sam/inference/pseudo_anno_regionclip.json

    if step == 3:
        dataset_name = 'ytvis_2021_dev'
        pred_file = 'output/video_sam/inference/pseudo_anno_regionclip.json'
        pred_results = json.load(open(pred_file, 'r'))
        gt_file = 'output/video_sam/inference/gt_anno_regionclip.json'
        gt_results = json.load(open(gt_file, 'r'))
        dataset_meta_file = 'datasets/ytvis_2021/valid_sub_0.01.json'
        dataset = json.load(open(dataset_meta_file, 'r'))
        evaluate_predictions_on_ytvis(dataset_name, pred_file)

    print('Done!')