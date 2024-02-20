import contextlib
import copy
import sys
import io
import itertools
import json
import logging
import numpy as np
import os
import torch

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import pycocotools.mask as mask_util

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, COCOEvaluator
from detectron2.utils.file_io import PathManager

from .davis2017_evaluation.davis2017.results import Results
from .davis2017_evaluation.davis2017.davis import DAVIS
from .davis2017_evaluation.davis2017.metrics import db_eval_boundary, db_eval_iou
from .davis2017_evaluation.davis2017.utils import db_statistics

# from davis2017_evaluation.davis2017.results import Results
# from davis2017_evaluation.davis2017.davis import DAVIS
# from davis2017_evaluation.davis2017.metrics import db_eval_boundary, db_eval_iou
# from davis2017_evaluation.davis2017.utils import db_statistics



class DAVISEvaluator(DatasetEvaluator):
    """
    Save the prediction results for RefVOS task, and evaluate the metrics: J, F, J&F (also termed as G^{th})
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
        gt_set='val',
        sequences='all',
        metrics=('J', 'F'),
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of 'semi-supervised', 'unsupervised'.
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

        if tasks is None or isinstance(tasks, CfgNode):
            self._logger.warning(
                "Referring Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            if 'refdavis' in dataset_name:
                self._tasks = 'unsupervised'
            elif 'davis16' in dataset_name or 'davis17' in dataset_name:
                self._tasks = 'semi-supervised'  # Infering it from predictions should be better
            else:
                ValueError
        else:
            self._tasks = tasks

        assert self._tasks is not None and isinstance(self._tasks, (str, list)), f"invalid type: {self._tasks}"
        self.task = self._tasks if isinstance(self._tasks, str) else self._tasks[0]
        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        if 'refdavis' in dataset_name:
            gt_root = PathManager.get_local_path('/'.join(self._metadata.image_root.split('/')[:-2] + ['DAVIS']))
            resolution = '480p'
        elif 'davis16' in dataset_name or 'davis17' in dataset_name:
            gt_root = PathManager.get_local_path('/'.join(self._metadata.image_root.split('/')[:-2]))
            resolution = 'Full-Resolution'
        else:
            ValueError
        
        self.dataset = DAVIS(
            root=os.path.join(os.getcwd(), gt_root), 
            task=self.task, 
            subset=gt_set, 
            sequences=sequences, 
            resolution=resolution
        )

        self.debug = False
        self.metrics = metrics if isinstance(metrics, tuple) or isinstance(metrics, list) else [metrics]
        if 'T' in self.metrics:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in self.metrics and 'F' not in self.metrics:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

    
    def reset(self):
        self._predictions = []
        PathManager.mkdirs(self._output_dir)

    def process(self, inputs, outputs):
        """
        save referring VOS segmentation result 
        """
    
    def evaluate(self):
        res_path = os.path.join(self._output_dir, 'Annotations')
        metrics_res = evaluate_davis(self.dataset, res_path, self.task, self.metrics)

        # save metrics
        output_file = open(os.path.join(self._output_dir, 'davis-metrics.txt'), 'w')
        for m, m_res in metrics_res.items():
            output_file.write(f'Saving metric {m}')
            for k, v in m_res.items():
                if k in {"M", "R", "D"}:
                    v = sum(v)/len(v) * 100.
                    output_file.write(f'{k} : {v}')
                    print("{}: {:.2f}".format(k, v))
        output_file.close() 

    
def evaluate_davis(dataset, res_path, task, metrics, debug=False):
    '''
    original code from https://github.com/davisvideochallenge/davis2017-evaluation/tree/master
    '''
    # Containers
    metrics_res = {}
    if 'J' in metrics:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metrics:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    # Sweep all sequences
    results = Results(root_dir=res_path)
    for seq in tqdm(list(dataset.get_sequences())):
        all_gt_masks, all_void_masks, all_masks_id = dataset.get_all_masks(seq, True)
        if task == 'semi-supervised':
            all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
        all_res_masks = results.read_masks(seq, all_masks_id)
        if task == 'unsupervised':
            j_metrics_res, f_metrics_res = _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metrics)
        elif task == 'semi-supervised':
            j_metrics_res, f_metrics_res = _evaluate_semisupervised(all_gt_masks, all_res_masks, None, metrics)

        for ii in range(all_gt_masks.shape[0]):
            seq_name = f'{seq}_{ii+1}'
            if 'J' in metrics:
                [JM, JR, JD] = db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
                metrics_res['J']["M_per_object"][seq_name] = JM
            if 'F' in metrics:
                [FM, FR, FD] = db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)
                metrics_res['F']["M_per_object"][seq_name] = FM

        # Show progress
        if debug:
            sys.stdout.write(seq + '\n')
            sys.stdout.flush()
    print('All metrics:', metrics_res)
    return metrics_res

def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metrics):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write(
            "\nIn your PNG files there is an index higher than the number of objects in the sequence!"
        )
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metrics:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metrics:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)

    return j_metrics_res, f_metrics_res

def _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metrics, max_n_proposals=20):
    if all_res_masks.shape[0] > max_n_proposals:
        sys.stdout.write(
            f"\nIn your PNG files there is an index higher than the maximum number ({max_n_proposals}) of proposals allowed!"
        )
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
    f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
    for ii in range(all_gt_masks.shape[0]):
        for jj in range(all_res_masks.shape[0]):
            if 'J' in metrics:
                j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
            if 'F' in metrics:
                f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
    if 'J' in metrics and 'F' in metrics:
        all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
    else:
        all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
    row_ind, col_ind = linear_sum_assignment(-all_metrics)

    return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]


if __name__ == "__main__":
    data_type = 'refer'
    if data_type == 'vos':
        task = 'semi-supervised'
        metrics = ('J', 'F')
        gt_root = "datasets/DAVIS" 
        resolution='Full-Resolution'
        res_path = os.path.join(
            os.getcwd(), 
            "output/v2/univs_swinb_f5_stage2_4x_c1+univs_frozenbb_category_only/inf_training/vos/davis_prompt_50k/inference/Annotations"
        )
        print(os.path.join(os.getcwd(), gt_root))
        dataset = DAVIS(
            root=os.path.join(os.getcwd(), gt_root), 
            task=task, 
            subset='val', 
            sequences='all',
            resolution=resolution,
        )
        evaluate_davis(dataset, res_path, task, metrics)

    elif data_type == 'refer':
        task = 'unsupervised'
        metrics = ('J', 'F')
        gt_root = "datasets/ref-davis/DAVIS"
        resolution='480p'
        res_path = os.path.join(
            os.getcwd(), 
            "output/univs_prompt_swinl_bs8_f4_video_1x_200queries_c1+univs+casa_prompt+sa_frozenbb_ftrefer+burst/inf_training/pvos/viposeg_dev/inference/Annotations"
        )
        print(os.path.join(os.getcwd(), gt_root))
        dataset = DAVIS(
            root=os.path.join(os.getcwd(), gt_root), 
            task=task, 
            subset='val', 
            sequences='all',
            resolution=resolution,
        )
        evaluate_davis(dataset, res_path, task, metrics)
    else:
        raise ValueError('Not implemented type!')