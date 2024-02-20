import torch
import json
import os

import matplotlib.pyplot as plt
import pycocotools.mask as mask_util

from detectron2.structures import Instances, Boxes
from detectron2.data import MetadataCatalog
from detectron2.utils.memory import retry_if_cuda_oom

from mdqe.util.misc import interpolate
from video_visualizer import VideoVisualizer


class ShowMasksFromJson:
    def __init__(self, res_file_dir=None, data_type='ovis'):
        self.res_file_dir = res_file_dir
        self.data_type = data_type
        self.anno_json_file = 'datasets/' + self.data_type + '/valid.json'
        self.imgs_dir = 'datasets/' + self.data_type + '/valid/JPEGImages'
        self.video_names = json.load(open(self.anno_json_file))['videos']

        if self.data_type == 'ovis':
            self.metadata = MetadataCatalog.get("ytvis_ovis_val")
        else:
            self.metadata = MetadataCatalog.get("ytvis_2021_val")
        self.VideoVisualizer = VideoVisualizer(self.metadata)

    def plot_img_with_masks(self, res_file_name):
        print('Loading and preparing results...')
        resList = json.load(open('/'.join([self.res_file_dir, res_file_name])))
        print('Finish loading results!')

        # Merge all instance results that belong to a same video into a Dict
        resDict = {res['video_id']: [] for res in resList}
        for res in resList:
            resDict[res['video_id']].append(res)
        del resList

        for vid_id in resDict.keys():
            reslist_vid = resDict[vid_id]
            if len(reslist_vid) == 0:
                continue

            masks, scores, labels = [], [], []
            for i, res in enumerate(reslist_vid):
                if res['score'] <= 0.1:
                    continue
                    
                scores.append(res['score'])
                labels.append(res['category_id'] - 1)

                _masks = torch.from_numpy(mask_util.decode(res['segmentations'])).permute(2, 0, 1).float().cuda()
                H, W = _masks.shape[-2] // 4, _masks.shape[-1] // 4
                if (H < 240 or W < 240) and _masks.shape[0] <= 100:
                    H, W = _masks.shape[-2] // 2, _masks.shape[-1] // 2
                _masks = retry_if_cuda_oom(interpolate)(
                    _masks.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                ).gt(0.5).float().squeeze(0)
                # visual segmented instance results with interval of 2 frames
                _masks = _masks[::2].cpu()  # T//2xHxW
                masks.append(_masks)

            if len(masks) == 0:
                continue

            scores, idx = torch.as_tensor(scores).sort(descending=True)
            labels = torch.as_tensor(labels)[idx]
            masks = torch.stack(masks, dim=0)[idx]  # NxTxHxW
            
            # Remove repeated masks by mask IoU, cause the model turns on is_multi_cls
            valid = [0]
            masks_sub = masks[:, ::max(masks.shape[1] // 25, 1)]
            for i, m in enumerate(masks_sub[1:]):
                numerator = masks_sub[:i+1] * m.unsqueeze(0)
                denominator = masks_sub[:i+1] + m.unsqueeze(0) - numerator
                siou = numerator.sum(dim=(-1, -2)) / denominator.sum(dim=(-1, -2)).clamp(min=1)
                if siou.mean() < 0.75 and siou.max() < 0.98:
                    valid.append(i+1)

            scores = scores[valid]
            labels = labels[valid]
            masks = masks[valid]

            # Visualize segmented instance masks by the builtin function 'visualizer.py' in d2
            saved_video_dir = '/'.join([self.res_file_dir, res_file_name[:-5], str(vid_id)])
            if not os.path.exists(saved_video_dir):
                os.makedirs(saved_video_dir)
            
            N, T, H, W = masks.shape
            file_names = self.video_names[vid_id - 1]['file_names']
            for t in range(masks.shape[1]):
                # Produce bounding boxes according to predicted masks
                pred_boxes = torch.zeros((N, 4), dtype=torch.float32, device=masks.device)
                x_any = torch.any(masks[:, t], dim=-2)
                y_any = torch.any(masks[:, t], dim=-1)
                for idx in range(masks[:, t].shape[0]):
                    x = torch.where(x_any[idx, :])[0]
                    y = torch.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        pred_boxes[idx, :] = torch.as_tensor(
                            [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32)

                result = Instances((H, W))
                result.scores = scores
                result.pred_classes = labels
                result.pred_masks = masks[:, t]
                result.pred_boxes = Boxes(pred_boxes)

                # Visualize segmented instance results with interval of 2 frames => t*2
                frame = plt.imread('/'.join([self.imgs_dir, file_names[t * 2]]))
                frame = retry_if_cuda_oom(interpolate)(
                    torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False
                ).squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()

                VisImage = self.VideoVisualizer.draw_instance_predictions(frame, result)
                VisImage.save('/'.join([saved_video_dir, str(t) + '.png']))


if __name__ == "__main__":

    res_file_dir = '/path/to/your/json/file/'
    data_type = 'ovis'  # must in {'ovis', 'ytvis_2019', 'ytvis_2021'}
    ShowRes = ShowMasksFromJson(res_file_dir=res_file_dir, data_type=data_type)
    ShowRes.plot_img_with_masks('results.json')