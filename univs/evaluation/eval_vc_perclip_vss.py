# Original code from https://github.com/sssdddwww2/CVPR2021_VSPW_Implement/tree/master

import argparse
import numpy as np
import os
from PIL import Image
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='VPSW eval')
    parser.add_argument('--submit_dir', '-i', type=str,
                        help='test output directory')

    parser.add_argument('--data_dir', type=str,
                        default="datasets/VSPW_480p/", help='/your/path/to/VSPW_480p')

    parser.add_argument('--split_file', type=str, default="val.txt",
                        help='Split file names, which can be val.txt, test.txt')

    args = parser.parse_args()
    return args


def map_category_id(gt_image):
    # Notice: Our ground truth mask contains values from 0 to 124 and 255. 
    # 0 indicates "others" and 255 indicates "void label". During evaluation, both 0 and 255 are void classes.
    gt_image[gt_image==0] = 255
    gt_image = gt_image - 1
    gt_image[gt_image==254] = 255
    return gt_image


def get_common(imglist_, predlist, clip_num, h, w):
    accs = []
    for i in range(len(imglist_) - clip_num):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))
                 
        for j in range(1, clip_num):
            # independent to categories
            common = (imglist_[i] == imglist_[i+j])
            global_common = np.logical_and(global_common, common)
            pred_common = (predlist[i] == predlist[i+j])
            predglobal_common = np.logical_and(predglobal_common, pred_common)
        pred = (predglobal_common * global_common)

        acc = pred.sum()/global_common.sum()
        accs.append(acc)

    return accs

        
def main():
    args = parse_args()
    DIR = args.data_dir
    Pred = args.submit_dir
    split = args.split_file
    output_dir = Pred

    with open(os.path.join(DIR, split),'r') as f:
        lines = f.readlines()
        videolist = [line[:-1] for line in lines]  # remove '\n'

    clip_nums = [8, 16]

    for clip_num in clip_nums:
        total_acc=[]
        for video in videolist:
            if video[0]=='.':
                continue
            imglist = []
            predlist = []

            images = sorted(os.listdir(os.path.join(DIR,'data', video, 'mask')))

            if len(images) <= clip_num:
                continue
            for imgname in images:
                if imgname[0]=='.':
                    continue
                img = Image.open(os.path.join(DIR, 'data', video, 'mask', imgname))
                w,h = img.size
                img = np.array(img)
                img = map_category_id(img)
                imglist.append(img)
                pred = Image.open(os.path.join(Pred, video, imgname))
                pred = np.array(pred)
                predlist.append(pred)
                
            accs = get_common(imglist, predlist, clip_num, h,w)
            # print('acc per video:', sum(accs)/len(accs))
            total_acc.extend(accs)

        total_acc = np.asarray(total_acc)
        Acc = np.nanmean(total_acc)
        print('*'*100)
        print('VC{} score: {} on {} set'.format(clip_num, Acc, split))
        print('*'*100)

        output_filename = os.path.join(output_dir, 'vc{}-final.txt'.format(clip_num))
        output_file = open(output_filename, 'w')
        output_file.write('VC{} score: {} on {} set'.format(clip_num, Acc, split))
        output_file.close() 

if __name__ == "__main__":
    main()
