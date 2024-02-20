import numpy as np
from glob import glob
import json
from tqdm import tqdm
from PIL import Image
import os.path as osp
import argparse

from eval_utils_viposeg import *


def eval_iou(res_list,seq_list,ref_list,obj_class_dict,eval_decay=False):
    # mask iou
    thing_seen_miou_list = []
    stuff_seen_miou_list = []
    thing_unseen_miou_list = []
    stuff_unseen_miou_list = []

    # boundary iou
    thing_seen_biou_list = []
    stuff_seen_biou_list = []
    thing_unseen_biou_list = []
    stuff_unseen_biou_list = []
    
    # decay
    if eval_decay:
        iou_decay_dict = {}
        for i in range(80):
            iou_decay_dict[i] = []
    vp = VIPOSeg()
       
    for s,r,f in tqdm(zip(seq_list,res_list,ref_list)):
        video_id = s.split('/')[-1]
        
        
        label_list = sorted(glob(s+'/*'))
        pred_list = sorted(glob(r+'/*'))
        ann_list = sorted(glob(f+'/*'))
        ann_name_list = [x.split('/')[-1] for x in ann_list]
            
        

        assert len(label_list) == len(pred_list), 'incomplete label/pred'
        
        obj_ids = []
        for i in range(len(label_list)):
                
            label = Image.open(label_list[i])
            pred = Image.open(pred_list[i])
            

            label = np.array(label,np.uint8)
            pred = np.array(pred,np.uint8)
            
            obj_num = len(obj_ids)
            for id in obj_ids:
                mask_gt = label==id
                mask_pred = pred==id
                
                # mask iou and boundary iou
                if (np.sum(mask_pred) == 0) and (np.sum(mask_gt) != 0):
                    miou = 0.
                    biou = 0.
                elif (np.sum(mask_pred) != 0) and (np.sum(mask_gt) == 0):
                    miou = 0.
                    biou = 0.
                elif (np.sum(mask_pred) ==0) and (np.sum(mask_gt) ==0):
                    miou = 1.
                    biou = 1.
                else:
                    miou = np.sum(mask_gt & mask_pred) / np.sum(mask_gt | mask_pred)
                    biou = boundary_iou(mask_gt.astype(np.uint8),mask_pred.astype(np.uint8),dilation_ratio=0.02)
                
                class_id = int(obj_class_dict[video_id][str(id)])
                if class_id == 98:
                    if video_id in vp.other_machine_videos:
                        stuff_unseen_miou_list.append(miou)
                        stuff_unseen_biou_list.append(biou)
                    else:
                        stuff_seen_miou_list.append(miou)
                        stuff_seen_biou_list.append(biou)
                elif class_id in vp.thing_unseen_class:
                    thing_unseen_miou_list.append(miou)
                    thing_unseen_biou_list.append(biou)
                elif class_id in vp.stuff_unseen_class:
                    stuff_unseen_miou_list.append(miou)
                    stuff_unseen_biou_list.append(biou)
                elif class_id in vp.thing_seen_class:
                    thing_seen_miou_list.append(miou)
                    thing_seen_biou_list.append(biou)
                elif class_id in vp.stuff_seen_class:
                    stuff_seen_miou_list.append(miou)
                    stuff_seen_biou_list.append(biou)
                if eval_decay:
                    iou_decay_dict[obj_num].append((miou+biou)/2.)
            
            # exclude obj in ref frames, eval in next frame
            frame_name = label_list[i].split('/')[-1]
            if frame_name in ann_name_list:
                ann_idx = ann_name_list.index(frame_name)
                ann = Image.open(ann_list[ann_idx])
                obj_ids.extend([x for x in np.unique(ann) if x!=0])

    res_dict = {}
    res_dict['thing_seen_miou'] = np.mean(thing_seen_miou_list)
    res_dict['thing_unseen_miou'] = np.mean(thing_unseen_miou_list)
    res_dict['stuff_seen_miou'] = np.mean(stuff_seen_miou_list)
    res_dict['stuff_unseen_miou'] = np.mean(stuff_unseen_miou_list)
    res_dict['thing_seen_biou'] = np.mean(thing_seen_biou_list)
    res_dict['thing_unseen_biou'] = np.mean(thing_unseen_biou_list)
    res_dict['stuff_seen_biou'] = np.mean(stuff_seen_biou_list)
    res_dict['stuff_unseen_biou'] = np.mean(stuff_unseen_biou_list)
    res_dict['thing_seen_iou'] = (res_dict['thing_seen_miou']+res_dict['thing_seen_biou'])/2
    res_dict['thing_unseen_iou'] = (res_dict['thing_unseen_miou']+res_dict['thing_unseen_biou'])/2
    res_dict['stuff_seen_iou'] = (res_dict['stuff_seen_miou']+res_dict['stuff_seen_biou'])/2
    res_dict['stuff_unseen_iou'] = (res_dict['stuff_unseen_miou']+res_dict['stuff_unseen_biou'])/2
    res_dict['overall_iou'] = (res_dict['thing_seen_iou']+res_dict['thing_unseen_iou']\
                                +res_dict['stuff_seen_iou']+res_dict['stuff_unseen_iou'])/4
    
    if eval_decay:
        x = []
        y = []
        for k in iou_decay_dict.keys():
            v = iou_decay_dict[k]
            if v!=[] and k<60:
                x.append(k)
                y.append(np.mean(v))           
        _x = np.expand_dims(np.array(x),-1)
        _y = np.expand_dims(np.array(y),-1)
        A = _x/100
        b = -np.log(_y)
        decay = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b)
        res_dict['decay'] = decay[0,0]
    return res_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='./VIPOSeg/valid')
    parser.add_argument('--res_path',type=str,required=True)
    parser.add_argument('--eval_decay',action='store_true')
    args = parser.parse_args()
    
    res_list = sorted(glob(args.res_path+'/*'))
    seq_list = sorted(glob(osp.join(args.data_path,'Annotations_gt')+'/*'))
    ref_list = sorted(glob(osp.join(args.data_path,'Annotations')+'/*'))

    if len(res_list) < len(ref_list):
        res_seqs = [seq_name.split('/')[-1] for seq_name in res_list]
        ref_list_new = []
        for seq_name in ref_list:
            if seq_name.split('/')[-1] in res_seqs:
                ref_list_new.append(seq_name)
        ref_list = ref_list_new
    
    assert len(res_list) > 0 and len(res_list) == len(res_list) and\
        len(res_list) == len(ref_list), "{} results and {} data".format(len(res_list),len(ref_list))
    # read obj_class.json
    with open(osp.join(args.data_path,'obj_class.json'),'r') as f:
        obj_class_dict = json.load(f)
    
    res_dict = eval_iou(res_list,seq_list,ref_list,obj_class_dict,args.eval_decay)
    for k in res_dict.keys():
        v = res_dict[k]*100 if 'iou' in k else res_dict[k]
        print("{}: {:.2f}".format(k,v))