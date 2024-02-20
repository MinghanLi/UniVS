import logging
import numpy as np
import os
import torch

from PIL import Image

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

from .eval_utils_vss import Evaluator


class VSSEvaluator(DatasetEvaluator):
    """
    Save the prediction results in VSPW format and evaluate the metrics: MIoU and VC8/VC16
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
        eval_miou_res=-1,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        self.ignore_val = self._metadata.ignore_label
        dataset_id_to_contiguous_id = self._metadata.stuff_dataset_id_to_contiguous_id
        self.contiguous_id_to_dataset_id = {}
        for i, key in enumerate(dataset_id_to_contiguous_id.keys()):
            self.contiguous_id_to_dataset_id.update({i: key})
        
        self.num_classes = len(self._metadata.stuff_classes)
        self.image_root = self._metadata.image_root
        self.split_txt = self._metadata.split_txt
        self.eval_miou_res = eval_miou_res

        self._do_evaluation = True

    def reset(self):
        self._predictions = []
        PathManager.mkdirs(self._output_dir)

    def process(self, inputs, outputs):
        """
         save semantic segmentation result as an image
        """
        assert len(inputs) == 1, "More than one inputs are loaded for inference!"

        video_id = str(inputs[0]["video_id"])  # video name
        image_names = [inputs[0]['file_names'][idx] for idx in inputs[0]["frame_indices"]]
        img_shape = outputs['image_size']
        sem_seg_result = outputs['pred_masks'].numpy().astype(np.uint8)  # (t, h, w)
        sem_seg_result_ = np.zeros_like(sem_seg_result, dtype=np.uint8) + 255
        unique_cls = np.unique(sem_seg_result)
        for cls_id in unique_cls:
            if cls_id == self.ignore_val:
                continue
            cls_ = self.contiguous_id_to_dataset_id[cls_id] - min(self.contiguous_id_to_dataset_id.values())
            sem_seg_result_[sem_seg_result == cls_id] = cls_
        
        sem_seg_result = sem_seg_result_
        assert len(image_names) == len(sem_seg_result), 'Mismatch length between predicted and gt images'
        for i, image_name in enumerate(image_names):
            image_ = Image.fromarray(sem_seg_result[i])
            if not os.path.exists(os.path.join(self._output_dir, video_id)):
                os.makedirs(os.path.join(self._output_dir, video_id))
            image_.save(os.path.join(self._output_dir, video_id, image_name.split('/')[-1].split('.')[0] + '.png'))
        return

    def evaluate(self):
        """
        evaluate miou and vc8/vc16
        """
        if self._do_evaluation and comm.get_rank() == 0:
            self.evaluate_miou()
            self.evaluate_vc_perclip()

        return {}
    
    def evaluate_miou(self):
        eval_ = Evaluator(self.num_classes)
        eval_.reset()

        data_dir = '/'.join(self.image_root.split('/')[:2])
        split = self.split_txt

        with open(os.path.join(data_dir, split),'r') as f:
            lines = f.readlines()
            videolist = [line[:-1] for line in lines]
        for video in videolist:
            for tar in os.listdir(os.path.join(data_dir,'data',video,'mask')):
                pred = os.path.join(self._output_dir, video, tar)
                tar_ = Image.open(os.path.join(data_dir,'data',video,'mask',tar))
                tar_ = np.array(tar_)
                tar_ = map_category_id(tar_)
                if self.eval_miou_res > 0:
                    tar_ = mmcv.imrescale(
                        img=tar_,
                        scale=(self.eval_miou_res, 100000),
                        return_scale=False,
                        interpolation='nearest',
                    )
                tar_ = tar_[np.newaxis,:]
                pred_ = Image.open(pred)
                pred_ = np.array(pred_)
                pred_ = pred_[np.newaxis,:]

                assert tar_.shape[-2:] == pred_.shape[-2:], 'Mismatch shapes between predicted and GT masks'
                eval_.add_batch(tar_,pred_)

        Acc = eval_.Pixel_Accuracy()
        Acc_class = eval_.Pixel_Accuracy_Class()
        mIoU = eval_.Mean_Intersection_over_Union()
        FWIoU = eval_.Frequency_Weighted_Intersection_over_Union()
        print('*'*100)
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('*'*100)

        output_dir = '/'.join(self._output_dir.split('/')[:-1])
        output_filename = os.path.join(output_dir, 'miou-final.txt')
        output_file = open(output_filename, 'w')
        output_file.write("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        output_file.close() 
    
    def evaluate_vc_perclip(self):
        data_dir = '/'.join(self.image_root.split('/')[:2])
        split = self.split_txt

        with open(os.path.join(data_dir, split),'r') as f:
            lines = f.readlines()
            for line in lines:
                videolist = [line[:-1] for line in lines]

        # mVC8, mVC16
        clip_nums = [8, 16]
        for clip_num in clip_nums:
            total_acc=[]
            for video in videolist:
                if video[0]=='.':
                    continue
                imglist = []
                predlist = []

                images = sorted(os.listdir(os.path.join(data_dir,'data', video, 'mask')))

                if len(images) <= clip_num:
                    continue
                for imgname in images:
                    if imgname[0]=='.':
                        continue
                    img = Image.open(os.path.join(data_dir, 'data', video, 'mask', imgname))
                    w,h = img.size
                    img = np.array(img)
                    img = map_category_id(img)
                    imglist.append(img)
                    pred = Image.open(os.path.join(self._output_dir, video, imgname))
                    pred = np.array(pred)
                    predlist.append(pred)
                    
                accs = get_common(imglist, predlist, clip_num, h,w)
                # print('acc per video:', sum(accs)/len(accs))
                total_acc.extend(accs)

            total_acc = np.array(total_acc)
            Acc = np.nanmean(total_acc)
            print('*'*100)
            print('VC{} score: {} on {} set'.format(clip_num, Acc, split))
            print('*'*100)

            output_dir = '/'.join(self._output_dir.split('/')[:-1])
            output_filename = os.path.join(output_dir, 'vc{}-final.txt'.format(clip_num))
            output_file = open(output_filename, 'w')
            output_file.write('VC{} score: {} on {} set'.format(clip_num, Acc, split))
            output_file.close() 

def map_category_id(gt_image):
    # Notice: Our ground truth mask contains values from 0 to 124 and 255. 
    # 0 indicates "others" and 255 indicates "void label". During evaluation, both 0 and 255 are void classes.
    gt_image[gt_image==0] = 255
    gt_image -= 1
    gt_image[gt_image==254] = 255
    return gt_image
    

def get_common(list_, predlist, clip_num, h, w):
    accs = []
    for i in range(len(list_) - clip_num):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))
                 
        for j in range(1, clip_num):
            common = (list_[i] == list_[i+j])
            global_common = np.logical_and(global_common, common)
            pred_common = (predlist[i] == predlist[i+j])
            predglobal_common = np.logical_and(predglobal_common, pred_common)
        pred = (predglobal_common * global_common)

        acc = pred.sum()/global_common.sum()
        accs.append(acc)

    return accs
