import itertools
import json
import time
import copy
import logging
import numpy as np
import os
import torch

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

from tqdm import tqdm
from PIL import Image
from panopticapi.utils import rgb2id
from panopticapi.utils import IdGenerator

from .eval_vpq_vps import vpq_compute_parallel
from .eval_stquality_vps import STQuality 


class VPSEvaluator(DatasetEvaluator):
    """
    Save the prediction results in VIPSeg format, and evaluate the metrics: VPQ and STQ
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
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
        thing_dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        stuff_dataset_id_to_contiguous_id = self._metadata.stuff_dataset_id_to_contiguous_id
        self.contiguous_id_to_thing_dataset_id = {}
        self.contiguous_id_to_stuff_dataset_id = {}
        for i, key in enumerate(thing_dataset_id_to_contiguous_id.values()):
            self.contiguous_id_to_thing_dataset_id.update({i: key})
        for i, key in enumerate(stuff_dataset_id_to_contiguous_id.values()):
            self.contiguous_id_to_stuff_dataset_id.update({i: key})

        self.categories = self._metadata.categories
        self.num_classes = len(self.categories)
        self.ignore_label = self._metadata.ignore_label

        # directory which contains panoptic annotation images in COCO format
        _root = os.getenv("DETECTRON2_DATASETS", "datasets")
        panoptic_root_dir = PathManager.get_local_path(self._metadata.panoptic_root)
        panoptic_json_file = PathManager.get_local_path(self._metadata.panoptic_json)
        self.pan_gt_json_file = os.path.join(_root, panoptic_json_file)
        self.truth_dir =  os.path.join(_root, panoptic_root_dir)

        if '_cocovid' in self.pan_gt_json_file:
            # must use original json file here
            self.pan_gt_json_file = self.pan_gt_json_file.replace('_cocovid', '')
       
        # evaluate vps metrics via evla_vpq_vspw.py or evla_vpq_stq_vspw.py
        self._do_evaluation = True

        self.num_processes = 8

    def reset(self):
        self._predictions = []
        PathManager.mkdirs(self._output_dir)
        if not os.path.exists(os.path.join(self._output_dir, 'pan_pred')):
            os.makedirs(os.path.join(self._output_dir, 'pan_pred'), exist_ok=True)

    def process(self, inputs, outputs):
        """
        save panoptic segmentation result as an image
        """
        assert len(inputs) == 1, "More than one inputs are loaded for inference!"
        color_generator = IdGenerator(self._metadata.categories)

        image_names = [inputs[0]['file_names'][idx] for idx in inputs[0]["frame_indices"]]
        # video_id = str(inputs[0]["video_id"])
        video_id = image_names[0].split('/')[-2]
        
        img_shape = outputs['image_size']
        pan_seg_result = outputs['pred_masks']
        segments_infos = outputs['segments_infos']
        segments_infos_ = []

        pan_format = np.zeros((pan_seg_result.shape[0], img_shape[0], img_shape[1], 3), dtype=np.uint8)
        for segments_info in segments_infos:
            id = segments_info['id']
            is_thing = segments_info['isthing']
            sem = segments_info['category_id']

            mask = pan_seg_result == id
            color = color_generator.get_color(sem)
            pan_format[mask] = color

            dts = []
            dt_ = {"category_id": int(sem)-1, "iscrowd": 0, "id": int(rgb2id(color))}
            for i in range(pan_format.shape[0]):
                area = mask[i].sum()
                index = np.where(mask[i].numpy())
                if len(index[0]) == 0:
                    dts.append(None)
                else:
                    if area == 0:
                        dts.append(None)
                    else:
                        x = index[1].min()
                        y = index[0].min()
                        width = index[1].max() - x
                        height = index[0].max() - y
                        dt = {"bbox": [x.item(), y.item(), width.item(), height.item()], "area": int(area)}
                        dt.update(dt_)
                        dts.append(dt)
            segments_infos_.append(dts)
        
        with open(self.pan_gt_json_file, 'r') as f:
            gt_jsons = json.load(f)

        categories = gt_jsons['categories']
        categories = {el['id']: el for el in categories}
            
        #### save image
        annotations = []
        for i, image_name in enumerate(image_names):
            image_ = Image.fromarray(pan_format[i])
            if not os.path.exists(os.path.join(self._output_dir, 'pan_pred', video_id)):
                os.makedirs(os.path.join(self._output_dir, 'pan_pred', video_id))
            image_.save(os.path.join(self._output_dir, 'pan_pred', video_id, image_name.split('/')[-1].split('.')[0] + '.png'))
            annotations.append({"segments_info": [item[i] for item in segments_infos_ if item[i] is not None], "file_name": image_name.split('/')[-1]})
        self._predictions.append({'annotations': annotations, 'video_id': video_id})

    def evaluate(self):
        """
        save jsons and comput vpq and stq metrics
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            file_path = os.path.join(self._output_dir, 'pred.json')
            with open(file_path, 'w') as f:
                json.dump({'annotations': predictions}, f)
        
        if self._do_evaluation:
            self.evaluate_vpq()
            self.evaluate_stq()

        return {}
    
    def evaluate_vpq(self):
        # modefied from DVIS (CVPR2023)
        if not os.path.isdir(self._output_dir):
            print("%s doesn't exist" % self._output_dir)
        if os.path.isdir(self._output_dir) and os.path.isdir(self.truth_dir):
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

        start_all = time.time()
        pan_pred_json_file = os.path.join(self._output_dir, 'pred.json')
        with open(pan_pred_json_file, 'r') as f:
            pred_jsons = json.load(f)
        with open(self.pan_gt_json_file, 'r') as f:
            gt_jsons = json.load(f)

        categories = gt_jsons['categories']
        categories = {el['id']: el for el in categories}
        # ==> pred_json, gt_json, categories

        start_time = time.time()

        pred_annos = pred_jsons['annotations']
        pred_j={}
        for p_a in pred_annos:
            pred_j[p_a['video_id']] = p_a['annotations']
        gt_annos = gt_jsons['annotations']
        gt_j  ={}
        for g_a in gt_annos:
            gt_j[g_a['video_id']] = g_a['annotations']

        gt_pred_split = []

        pbar = tqdm(gt_jsons['videos'])
        for video_images in pbar:
            pbar.set_description(video_images['video_id'])
        
            video_id = video_images['video_id']
            gt_image_jsons = video_images['images']
            if video_id not in pred_j:
                print(f"{video_id} does not in prediced json, please double check!!")
                continue
            gt_js = gt_j[video_id]
            pred_js = pred_j[video_id]
            assert len(gt_js) == len(pred_js)
            
            gt_pans =[]
            pred_pans = []
            for imgname_j in gt_image_jsons:
                imgname = imgname_j['file_name']

                pred_pans.append(os.path.join(self._output_dir, 'pan_pred', video_id, imgname))
                gt_pans.append(os.path.join(self.truth_dir, video_id,imgname))
                
            gt_pred_split.append(list(zip(gt_js,pred_js,gt_pans,pred_pans,gt_image_jsons)))
            # print('processing video:{}'.format(video_id))

        start_time = time.time()
        vpq_all, vpq_thing, vpq_stuff = [], [], []

        # for k in [0,5,10,15] --> num_frames_w_gt [1,2,3,4]
        for nframes in [1, 2, 4, 6, 8]:
            gt_pred_split_ = copy.deepcopy(gt_pred_split)
            # vpq_all_, vpq_thing_, vpq_stuff_ = vpq_compute(
            #         gt_pred_split_, categories, nframes, self._output_dir)
            vpq_all_, vpq_thing_, vpq_stuff_ = vpq_compute_parallel(
                gt_pred_split_, categories, nframes, self._output_dir, self.num_processes
            )

            del gt_pred_split_
            print(vpq_all_, vpq_thing_, vpq_stuff_)
            vpq_all.append(vpq_all_)
            vpq_thing.append(vpq_thing_)
            vpq_stuff.append(vpq_stuff_)

        output_filename = os.path.join(self._output_dir, 'vpq-final.txt')
        output_file = open(output_filename, 'w')
        output_file.write("vpq_all:%.4f\n"%(sum(vpq_all)/len(vpq_all)))
        output_file.write("vpq_thing:%.4f\n"%(sum(vpq_thing)/len(vpq_thing)))
        output_file.write("vpq_stuff:%.4f\n"%(sum(vpq_stuff)/len(vpq_stuff)))
        output_file.close()
        print("vpq_all:%.4f\n"%(sum(vpq_all)/len(vpq_all)))
        print("vpq_thing:%.4f\n"%(sum(vpq_thing)/len(vpq_thing)))
        print("vpq_stuff:%.4f\n"%(sum(vpq_stuff)/len(vpq_stuff)))
        print('==> All:', time.time() - start_all, 'sec')
    
    def evaluate_stq(self):
        # modefied from DVIS (CVPR2023)
        bit_shit = 16

        if not os.path.isdir(self._output_dir):
            print("%s doesn't exist" % self._output_dir)
        if os.path.isdir(self._output_dir) and os.path.isdir(self.truth_dir):
            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

        start_all = time.time()
        pan_pred_json_file = os.path.join(self._output_dir, 'pred.json')
        with open(pan_pred_json_file, 'r') as f:
            pred_jsons = json.load(f)
        with open(self.pan_gt_json_file, 'r') as f:
            gt_jsons = json.load(f)

        categories = gt_jsons['categories']

        thing_list_ = []
        for cate_ in categories:
            cat_id = cate_['id']
            isthing = cate_['isthing']
            if isthing:
                thing_list_.append(cat_id)

        stq_metric = STQuality(
            self.num_classes, thing_list_, self.ignore_label, bit_shit, 2**24
        )

        pred_annos = pred_jsons['annotations']
        pred_j={}
        for p_a in pred_annos:
            pred_j[p_a['video_id']] = p_a['annotations']
        gt_annos = gt_jsons['annotations']
        gt_j  ={}
        for g_a in gt_annos:
            gt_j[g_a['video_id']] = g_a['annotations']
        
        gt_pred_split = []

        pbar = tqdm(gt_jsons['videos'])
        for seq_id, video_images in enumerate(pbar):
            video_id = video_images['video_id']
            pbar.set_description(video_id)

            # print('processing video:{}'.format(video_id))
            gt_image_jsons = video_images['images']
            gt_js = gt_j[video_id]
            pred_js = pred_j[video_id]
            assert len(gt_js) == len(pred_js)
        
            gt_pans =[]
            pred_pans = []
            for imgname_j in gt_image_jsons:
                imgname = imgname_j['file_name']
                pred_pan = Image.open(os.path.join(self._output_dir, 'pan_pred', video_id,imgname))
                pred_pans.append(np.array(pred_pan))
                gt_pan = Image.open(os.path.join(self.truth_dir, video_id,imgname))
                if gt_pan.size != pred_pan.size:
                    # print(f"Dismatch shape betweem GT and pred masks: {gt_pan.size} and {pred_pan.size}, Resize ....")
                    gt_pan = gt_pan.resize(pred_pan.size, Image.NEAREST)
                gt_pans.append(np.array(gt_pan))
            gt_id_to_ins_num_dic={}
            list_tmp = []
            for segm in gt_js:
                for img_info in segm['segments_info']:
                    id_tmp_ = img_info['id']
                    if id_tmp_ not in list_tmp:
                        list_tmp.append(id_tmp_)
            for ii, id_tmp_ in enumerate(list_tmp):
                gt_id_to_ins_num_dic[id_tmp_]=ii
                
            pred_id_to_ins_num_dic={}
            list_tmp = []
            for segm in pred_js:
                for img_info in segm['segments_info']:
                    id_tmp_ = img_info['id']
                    if id_tmp_ not in list_tmp:
                        list_tmp.append(id_tmp_)
            for ii, id_tmp_ in enumerate(list_tmp):
                pred_id_to_ins_num_dic[id_tmp_]=ii

            for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(list(zip(gt_js,pred_js,gt_pans,pred_pans,gt_image_jsons))):
                #### Step1. Collect frame-level pan_gt, pan_pred, etc.
                gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
                pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
                pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256

                ground_truth_instance = np.ones_like(pan_gt)*255
                ground_truth_semantic = np.ones_like(pan_gt)*255
                for el in gt_json['segments_info']:
                    id_ = el['id']
                    cate_id = el['category_id']
                    ground_truth_semantic[pan_gt==id_] = cate_id
                    ground_truth_instance[pan_gt==id_] = gt_id_to_ins_num_dic[id_]

                ground_truth = ((ground_truth_semantic << bit_shit) + ground_truth_instance)

                prediction_instance = np.ones_like(pan_pred)*255
                prediction_semantic = np.ones_like(pan_pred)*255

                for el in pred_json['segments_info']:
                    id_ = el['id']
                    cate_id = el['category_id']
                    prediction_semantic[pan_pred==id_] = cate_id
                    prediction_instance[pan_pred==id_] = pred_id_to_ins_num_dic[id_]
                prediction = ((prediction_semantic << bit_shit) + prediction_instance)  

                stq_metric.update_state(ground_truth.astype(dtype=np.int32),
                                        prediction.astype(dtype=np.int32), seq_id) 
        result = stq_metric.result()   
        output_filename = os.path.join(self._output_dir, 'stq-final.txt')
        output_file = open(output_filename, 'w')
        output_file.write('STQ : {}'.format(result['STQ']))
        output_file.write('AQ :{}'.format(result['AQ']) )
        output_file.write('IoU:{}'.format(result['IoU']))
        output_file.close()      
        print('*'*100)
        print('STQ : {}'.format(result['STQ']))
        print('AQ :{}'.format(result['AQ']) )
        print('IoU:{}'.format(result['IoU']))
        # print('STQ_per_seq')
        # print(result['STQ_per_seq'])
        # print('AQ_per_seq')
        # print(result['AQ_per_seq'])
        # print('ID_per_seq')
        # print(result['ID_per_seq'])
        # print('Length_per_seq')
        # print(result['Length_per_seq'])
        # print('*'*100)
