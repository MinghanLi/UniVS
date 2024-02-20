import argparse
import torch
import json
import os

import pycocotools.mask as mask_util
from detectron2.utils.memory import retry_if_cuda_oom

from misc import interpolate

from detectron2.structures import Instances, Boxes
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt


class ShowMasksFromJsonVideo:
    def __init__(self, data_type='ovis'):
        self.data_type = data_type
        _root = os.getenv("DETECTRON2_DATASETS", "datasets")

        if data_type == 'ovis':
            self.metadata = MetadataCatalog.get("ovis_val")
        else:
            self.metadata = MetadataCatalog.get("ytvis_2021_val")
        self.VideoVisualizer = VideoVisualizer(self.metadata)

    def plot_img_with_masks(self, input_resFile, output_dir, conf_threshold):
        print('Loading and preparing results...')
        resList = json.load(open(input_resFile))
        print('Finish loading results!')

        resVidIds = [res['video_id'] for res in resList]

        video_names = json.load(open('datasets/' + self.data_type + '/valid.json'))['videos']

        resDict = {vid_id: [] for vid_id in resVidIds}
        for res in resList:
            resDict[res['video_id']].append(res)
        del resList

        for vid_id in resDict.keys():
            if self.data_type == 'ovis' and vid_id == 2:
                # skip the video with around 300 frames
                continue

            file_names = video_names[vid_id - 1]['file_names']
            reslist_vid = resDict[vid_id]
            if len(reslist_vid) == 0:
                continue

            masks, scores, labels = [], [], []
            for i, res in enumerate(reslist_vid):
                if res['score'] <= conf_threshold:
                    continue

                scores.append(res['score'])
                labels.append(res['category_id']-1)

                _masks = torch.from_numpy(mask_util.decode(res['segmentations'])).permute(2, 0, 1).float().cuda()
                h, w = _masks.shape[-2] // 4, _masks.shape[-1] // 4
                if (h < 240 or w < 240) and _masks.shape[0] <= 480:
                    h, w = _masks.shape[-2] // 2, _masks.shape[-1] // 2
                _masks = retry_if_cuda_oom(interpolate)(
                    _masks.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
                ).gt(0.5).float().squeeze(0)
                masks.append(_masks.cpu())  # [TxHxW]

            if len(masks) == 0:
                continue

            scores, idx = torch.as_tensor(scores).sort(descending=True)
            labels = torch.as_tensor(labels)[idx]
            masks = torch.stack(masks, dim=0)[idx]  # NxTxHxW

            valid = [0]
            _masks = masks[:, ::max(masks.shape[1]//15, 1)]
            for i, m in enumerate(_masks):
                # remove repeated instance masks with multi_cls_on
                if i > 1:
                    numerator = _masks[:i] * m.unsqueeze(0)
                    denominator = _masks[:i] + m.unsqueeze(0) - numerator
                    siou = numerator.sum(dim=(-1, -2)) / denominator.sum(dim=(-1, -2)).clamp(min=1)
                    if siou.mean() < 0.6:
                        valid.append(i)

            scores = scores[valid]
            labels = labels[valid]
            masks = masks[valid]

            for t in range(masks.shape[1]):
                # Produce bounding boxes according to predicted masks
                pred_boxes = torch.zeros((masks[:, t].shape[0], 4), dtype=torch.float32, device=masks.device)
                x_any = torch.any(masks[:, t], dim=-2)
                y_any = torch.any(masks[:, t], dim=-1)
                for idx in range(masks[:, t].shape[0]):
                    x = torch.where(x_any[idx, :])[0]
                    y = torch.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        pred_boxes[idx, :] = torch.as_tensor(
                            [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32)

                result = Instances((h, w))
                result.scores = scores
                result.pred_classes = labels
                result.pred_masks = masks[:, t]
                result.pred_boxes = Boxes(pred_boxes)

                frame = plt.imread('/'.join(['datasets', self.data_type, 'valid/JPEGImages', file_names[t]]))
                frame = retry_if_cuda_oom(interpolate)(
                    torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0), size=(h, w),
                    mode="bilinear", align_corners=False
                ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

                VisImage = self.VideoVisualizer.draw_instance_predictions(frame, result)

                saved_results_dir = '/'.join([output_dir, input_resFile.split("/")[-1][:-5], str(vid_id)])
                if not os.path.exists(saved_results_dir):
                    os.makedirs(saved_results_dir)
                    print('Segmented masks are saved in: ', saved_results_dir)

                saved_results_path = '/'.join([saved_results_dir, str(t) + '.png'])
                VisImage.save(saved_results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="ovis")
    parser.add_argument("--conf-threshold", default=0.2, type=float, help="confidence threshold")
    args = parser.parse_args()

    assert args.dataset in {'ovis', 'ytvis_2021', 'ytvis_2019'}
    ShowRes = ShowMasksFromJson(data_type=args.dataset)
    ShowRes.plot_img_with_masks(args.input, args.output, args.conf_threshold)