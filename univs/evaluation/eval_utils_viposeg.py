import cv2
import numpy as np

class VIPOSeg:
        thing_class = [60, 89, 90, 8, 48, 2, 79, 106, 76, 84, 114, 74, 108, 91, 83, 85, 54, 65, 78, 44, 
                    92, 122, 107, 43, 88, 117, 50, 51, 87, 52, 62, 115, 10, 41, 77, 82, 56, 123, 49, 
                    4, 63, 102, 99, 109, 47, 55, 61, 118, 72, 46, 96, 64, 101, 86, 97, 100, 116, 95]
        stuff_class = [28, 66, 0, 14, 15, 13, 7, 12, 22, 68, 1, 59, 27, 75, 40, 29, 18, 21, 19, 39, 30, 
                    11, 53, 111, 45, 35, 98, 36, 119, 42, 104, 23, 80, 93, 67, 3, 31, 16, 69, 103, 37, 
                    121, 110, 105, 33, 24, 70, 73, 32, 9, 71, 120, 58, 94, 5, 34, 20, 6]

        thing_unseen_class = [102, 99, 109, 47, 55, 61, 118, 72, 46, 96, 64, 101, 86, 97, 100, 116, 95]
        stuff_unseen_class = [9, 71, 120, 58, 94, 5, 34, 20, 6, 26, 112, 17, 57, 113, 25, 81, 38]
        

        other_machine_cl = 98
        other_machine_videos = ['187_WUZUSD4477I', '319_l1Dz12fxQzQ', '320_nhKXemkIvh4', '517_AWvYuplla_s', 
                                '532_QmZyJuLlEec', '774_devdFjIpDcc', '1016_HG0AsTOxI5g', '1017_IAU0WGB9VPw', 
                                '1020_TgCIv6bp3XM', '1021_cPOxAMo28yk', '1022_emSaDd2ddj0', '1033_sh81AwYuihg', 
                                '1065_d2sHRyAHKqI', '1067_fk3jhxBi1pA', '1068_gxnZkf0LQfk', '1069_jFHRbZxswz8', 
                                '1070_uTJB31tuYes', '1072_zvNEdUk5k0Q', '1230_AGY-gQ_3O8Y', '1333__iprMPKLdOQ', 
                                '1334_qlmfvYA3_rk', '2004_1btxeVbyojs', '2005_83KrhWajwfw']
        def __init__(self) -> None:
            self.thing_seen_class = [x for x in self.thing_class if x not in self.thing_unseen_class]
            self.stuff_seen_class = [x for x in self.stuff_class if x not in self.stuff_unseen_class]

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_f(gt, dt, dilation_ratio=0.008):
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    tp = ((gt_boundary * dt_boundary) > 0).sum()
    p = (dt_boundary > 0).sum()
    t = (gt_boundary > 0).sum()
    if p==0:
        precision = 0
    else:
        precision = tp/p
    recall = tp/t
    if (recall+precision) == 0:
        f = 0
    else:
        f = 2*precision*recall / (precision+recall)
    return f

def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    if union == 0:
        boundary_iou = 0
    else:
        boundary_iou = intersection / union
    return boundary_iou