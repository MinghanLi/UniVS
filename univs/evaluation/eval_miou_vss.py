# Code is modefied from https://github.com/lxtGH/Tube-Link/blob/main/scripts/test_vspw/iou_cal.py

import argparse
import numpy as np
import os
from PIL import Image
import sys

from eval_utils_vss import Evaluator

eval_ = Evaluator(124)
eval_.reset()

def parse_args():
    parser = argparse.ArgumentParser(description='No description.')
    parser.add_argument('--data_dir', type=str, default='datasets/VSPW_480p/')
    parser.add_argument('--submit_dir', '-i', type=str,
                        help='test output directory')
    parser.add_argument('--split_file', type=str, default="val.txt",
                        help='Split file names, which can be val.txt, test.txt')
    parser.add_argument('--eval-res', type=int, default=-1)
    args = parser.parse_args()
    return args

def map_category_id(gt_image):
    # Notice: Our ground truth mask contains values from 0 to 124 and 255. 
    # 0 indicates "others" and 255 indicates "void label". During evaluation, both 0 and 255 are void classes.
    gt_image[gt_image==0] = 255
    gt_image -= 1
    gt_image[gt_image==254] = 255
    
    return gt_image


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.data_dir
    submit_dir = args.submit_dir
    split = args.split_file

    eval_res = args.eval_res

    with open(os.path.join(data_dir, split),'r') as f:
        lines = f.readlines()
        for line in lines:
            videolist = [line[:-1] for line in lines]

    for video in videolist:
        for tar in os.listdir(os.path.join(data_dir,'data',video,'mask')):
            pred = os.path.join(submit_dir,video,tar)
            tar_ = Image.open(os.path.join(data_dir,'data',video,'mask',tar))
            tar_ = np.array(tar_)
            tar_ = map_category_id(tar_)
            
            if eval_res > 0:
                # not used here
                import mmcv
                tar_ = mmcv.imrescale(
                    img=tar_,
                    scale=(eval_res, 100000),
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

    output_filename = os.path.join(submit_dir, 'miou-final.txt')
    output_file = open(output_filename, 'w')
    output_file.write("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    output_file.close()