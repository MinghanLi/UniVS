# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    '''
    The boxes should be in [x0, y0, x1, y1] format
    '''
    return torch.prod(boxes[..., 2:] - boxes[..., :2], dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)  # [B,N]
    area2 = box_area(boxes2)  # [B,M]

    lt = torch.max(boxes1[..., None, :2], boxes2[..., None, :, :2])  # [B,N,M,2]
    rb = torch.min(boxes1[..., None, 2:], boxes2[..., None, :, 2:])  # [B,N,M,2]

    wh = (rb - lt).clamp(min=0)        # [B,N,M,2]
    inter = torch.prod(wh, dim=-1)     # [B,N,M]

    union = (area1.unsqueeze(-1) + area2.unsqueeze(-2) - inter).clamp(min=1e-5)

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)

    out: [N, M]
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[..., 2:] >= boxes1[..., :2]).all()
    assert (boxes2[..., 2:] >= boxes2[..., :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[..., None, :2], boxes2[..., None, :, :2])
    rb = torch.max(boxes1[..., None, 2:], boxes2[..., None, :, 2:])

    wh = (rb - lt).clamp(min=0)    # [B,N,M,2]
    area = torch.prod(wh, dim=-1)  # [B,N,M]

    return iou - (area - union) / area.clamp(min=1e-5)


def video_box_l1(out_bbox, tgt_bbox, valid):
    """
    out_bbox: [N, num_frames, 4]
    tgt_bbox: [M, num_frames, 4]
    valid: [M, num_frames]

    out: [N, M]
    """
    # NOTE subtraction with abs leads to slight difference with torch.cdist.
    # However, as it is very subtle (nearly 1e-7), believe this would not harm the performance
    out_bbox = out_bbox.unsqueeze(1)    # [N, 1, num_frames, 4]
    tgt_bbox = tgt_bbox.unsqueeze(0)    # [1, M, num_frames, 4]

    cost_bbox = torch.abs(out_bbox - tgt_bbox).sum(dim=-1)
    cost_bbox[:, ~valid] = 0.0

    return cost_bbox.sum(dim=-1) / valid.float().sum(dim=-1)


def video_box_iou(boxes1, boxes2):
    """
    boxes1: [N, num_frames, 4]
    boxes2: [M, num_frames, 4]

    out: [N, M, numframes]
    """
    num_frames = boxes1.shape[1]
    N, M = len(boxes1), len(boxes2)

    area1 = box_area(boxes1.flatten(0,1)).view(N, num_frames)   # [N, num_frames]
    area2 = box_area(boxes2.flatten(0,1)).view(M, num_frames)   # [M, num_frames]

    lt = torch.max(boxes1[:, None, :, :2], boxes2[:, :, :2])    # [N, M, num_frames, 2]
    rb = torch.min(boxes1[:, None, :, 2:], boxes2[:, :, 2:])    # [N, M, num_frames, 2]

    wh = (rb - lt).clamp(min=0.)              # [N, M, num_frames, 2]
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [N, M, num_frames]

    union = area1[:, None, :] + area2 - inter
    
    iou = inter / union.clamp(min=1e-5)
    return iou, inter, union


def video_generalized_box_iou(out_bbox, tgt_bbox, valid=None):
    """
    out_bbox: [N, num_frames, 4]
    tgt_bbox: [M, num_frames, 4]
    valid: [M, num_frames]

    out: [N, M]
    """
    assert (out_bbox[..., 2:] >= out_bbox[..., :2]).all(), out_bbox
    assert (tgt_bbox[..., 2:] >= tgt_bbox[..., :2]).all(), tgt_bbox
    iou, inter, union = video_box_iou(out_bbox, tgt_bbox)  # [N, M, num_frames] respectively

    lt = torch.min(out_bbox[:, None, :, :2], tgt_bbox[None, :, :, :2])
    rb = torch.max(out_bbox[:, None, :, 2:], tgt_bbox[None, :, :, 2:])

    wh = (rb - lt).clamp(min=0)  # [N, M, num_frames, 2]
    area = wh[:, :, :, 0] * wh[:, :, :, 1]  # [N, M, num_frames]

    # handle empty boxes, [N, M, num_frames]
    giou = torch.where(
        inter > 0,
        iou - (area - union) / area,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    if valid is not None:
        giou[:, ~valid] = 0.0
        return giou.sum(dim=-1) / torch.clamp(valid.float().sum(dim=-1), min=1)
    else:
        return giou.mean(dim=-1)


def matched_boxlist_giou(boxes1, boxes2) -> torch.Tensor:
    """
    Compute pairwise generalized intersection over union (GIOU) of two sets of matched
    boxes. The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix

    Args:
        boxes1:  bounding boxes, sized [N,4].
        boxes2: bounding boxes, sized [N,4].
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(boxes2), "boxlists should have the same" "number of entries, " \
                                       "got {}, {}".format(len(boxes1), len(boxes2))
    assert (boxes1[..., 2:] >= boxes1[..., :2]).all(), boxes1
    assert (boxes2[..., 2:] >= boxes2[..., :2]).all(), boxes2

    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [N]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0.)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    union = (area1 + area2 - inter).clamp(min=1e-5)

    lt_cir = torch.min(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb_cir = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]
    wh_cir = (rb_cir - lt_cir).clamp(min=0.)  # [N,2]
    area = (wh_cir[:, 0] * wh_cir[:, 1]).clamp(min=1e-5)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / union - (area - union) / area,
        torch.zeros(1, dtype=union.dtype, device=inter.device),
        )

    return iou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e5).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e5).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


@torch.jit.script
def encode(matched, priors):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.

    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4] [x,y,x2,y2]
        - priors:  The tensor of all priors with shape [num_priors, 4] [cx,cy,w,h]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """

    variances = [0.1, 0.2]
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2. - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    loc = torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

    return loc


def decode(loc, priors):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)

    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.

    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).

    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4] [cx, cy, w, h]

    Returns: A tensor of decoded relative coordinates in point form
             with size [num_priors, 4]
    """

    variances = [0.1, 0.2]
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    # [x1, y1, x2, y2]
    return boxes


def box_frames_to_clip(boxes):
    """
    Args: boxes: [..., T, 4] with [x,y,x,y] format
    return: [..., 4] with [x,y,x,y] format
    """

    valid = ((boxes[..., 2:] - boxes[..., :2]) > 0).all(dim=-1).unsqueeze(dim=-1)  # ...xNxT
    circum_boxes = torch.cat([torch.where(valid, boxes[..., :2], 100*torch.ones_like(boxes[..., :2])).min(dim=-2)[0],
                              torch.where(valid, boxes[..., 2:], -100*torch.ones_like(boxes[..., :2])).max(dim=-2)[0]], dim=-1)
    cond = (circum_boxes != 100) & (circum_boxes != -100)  # ...xNx4
    circum_boxes = torch.where(cond, circum_boxes, torch.zeros_like(circum_boxes))

    return circum_boxes
