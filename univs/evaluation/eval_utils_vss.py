# Original code from https://github.com/VSPW-dataset/VSPW_code/blob/main/utils.py

import sys
import torch
import torch.nn as nn
import os
import logging
import re
import functools
import fnmatch
import numpy as np


def flowwarp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid,align_corners=False)

    return output

def get_common(list_,predlist,clip_num,h,w):
    accs = []
    for i in range(len(list_)-clip_num):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))


        for j in range(1,clip_num):
            common = (list_[i] == list_[i+j])
            global_common = np.logical_and(global_common,common)
            pred_common = (predlist[i]==predlist[i+j])
            predglobal_common = np.logical_and(predglobal_common,pred_common)
        pred = (predglobal_common*global_common)

        acc = pred.sum()/global_common.sum()
        accs.append(acc)
    return accs

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def beforeval(self):
        isval = np.sum(self.confusion_matrix,axis=1)>0
        self.confusion_matrix = self.confusion_matrix*isval

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc


    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        isval = np.sum(self.confusion_matrix,axis=1)>0
        print(MIoU, isval.sum())
        MIoU = np.nansum(MIoU*isval)/isval.sum()
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # if max(gt_image[mask]) == self.num_class:
        #     print('gt', mask.shape, np.unique(gt_image[mask]))
        #     print('pd', pre_image.shape, np.unique(pre_image[mask]))
        #     exit()
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def setup_logger(distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            if filename[0]=='.':
                continue
            files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret