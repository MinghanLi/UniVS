import torch
import torchvision
from torch import Tensor

def convert_box_to_mask(outputs_box: torch.Tensor, h: int, w: int):
    """
    Args:
        outputs_box: normalized boxes with the format [x1, y1, x2, y2]
        h: the height of output masks
        w: the width of output masks

    Returns:

    """
    box_shape = outputs_box.shape

    box_normalizer = torch.as_tensor([w, h, w, h], dtype=outputs_box.dtype,
                                     device=outputs_box.device).reshape(1, -1)

    outputs_box = outputs_box.flatten(0, -2)
    outputs_box_wonorm = outputs_box * box_normalizer  # ..., 4
    outputs_box_wonorm = torch.cat([outputs_box_wonorm[..., :2].floor(),
                                    outputs_box_wonorm[..., 2:].ceil()], dim=-1)
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=outputs_box.device),
                                    torch.arange(w, device=outputs_box.device))  # H, W
    grid_y = grid_y.reshape(1, h, w)
    grid_x = grid_x.reshape(1, h, w)

    # repeat operation will greatly expand the computational graph
    gt_x1 = grid_x > outputs_box_wonorm[..., 0, None, None]
    lt_x2 = grid_x <= outputs_box_wonorm[..., 2, None, None]
    gt_y1 = grid_y > outputs_box_wonorm[..., 1, None, None]
    lt_y2 = grid_y <= outputs_box_wonorm[..., 3, None, None]
    cropped_box_mask = gt_x1 & lt_x2 & gt_y1 & lt_y2

    cropped_box_mask = cropped_box_mask.reshape((*box_shape[:-1], h, w))

    return cropped_box_mask

def convert_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out

def calculate_mask_quality_scores(mask_pred, threshold=1):
    # mask_pred is the logits, before activation
    scores_mask = (mask_pred > threshold).flatten(1).sum(-1) / (mask_pred > -threshold).flatten(1).sum(-1).clamp(min=1)
    return scores_mask

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


def box_xyxy_to_xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    '''
    The boxes should be in [x0, y0, x1, y1] format
    '''
    return torch.prod(boxes[..., 2:] - boxes[..., :2], dim=-1)

def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4]
    boxes2: [M, 4]
    out: [N, M]
    """
    area1 = box_area(boxes1)  # N
    area2 = box_area(boxes2)  # M

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[..., 0] * wh[..., 1]  # [N, M]

    union = (area1[:, None] + area2[None] - inter).clamp(min=1e-3)

    iou = inter / union
    return iou, inter, union

def video_box_iou(boxes1, boxes2):
    """
    boxes1: [N, num_frames, 4]
    boxes2: [M, num_frames, 4]
    out: [N, M, num_frames]
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    area1 = box_area(boxes1)  # [N, num_frames]
    area2 = box_area(boxes2)  # [M, num_frames]

    lt = torch.max(boxes1[:, None, :, :2], boxes2[None, :, :, :2])  # [N, M, num_frames, 2]
    rb = torch.min(boxes1[:, None, :, 2:], boxes2[None, :, :, 2:])  # [N, M, num_frames, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, num_frames, 2]
    inter = wh[..., 0] * wh[..., 1]  # [N, M, num_frames]

    union = (area1[:, None] + area2[None] - inter).clamp(min=1e-3)

    iou = inter / union
    return iou, inter, union


def batched_box_iou(boxes1, boxes2):
    """
    boxes1: [B, N, 4]
    boxes2: [B, M, 4]
    out: [B, N, M]
    """
    area1 = box_area(boxes1)  # [B, N]
    area2 = box_area(boxes2)  # [B, M]

    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])  # [B, N, M, 2]
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])  # [B, N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [B, N, M, 2]
    inter = wh[..., 0] * wh[..., 1]  # [B, N, M]

    union = (area1[:, :, None] + area2[:, None] - inter).clamp(min=1e-3)

    iou = inter / union
    return iou, inter, union

def mask_iou(masks1, masks2):
    """
    masks1: [N, H, W] 
    masks2: [M, H, W]
    out: [N, M]
    """
    masks1 = masks1.flatten(-2).float()
    masks2 = masks2.flatten(-2).float()

    inter = ((masks1[:, None] + masks2[None]) == 2).sum(-1)
    union = ((masks1[:, None] + masks2[None]) >= 1).sum(-1).clamp(min=1)

    iou = inter / union

    return iou

def batched_mask_iou(masks1, masks2):
    """
    masks1: [B, N, H, W] 
    masks2: [B, M, H, W]
    out: [B, N, M]
    """
    masks1 = masks1.flatten(-2).float()
    masks2 = masks2.flatten(-2).float()

    inter = ((masks1[:, :, None] + masks2[:, None]) == 2).sum(-1)
    union = ((masks1[:, :, None] + masks2[:, None]) >= 1).sum(-1).clamp(min=1)

    iou = inter / union

    return iou

def batched_pair_mask_iou(masks1, masks2):
    """
    masks1: [B, N, H, W] 
    masks2: [B, N, H, W]
    out: [B, N]
    """
    masks1 = masks1.flatten(-2).float()
    masks2 = masks2.flatten(-2).float()

    inter = ((masks1 + masks2) == 2).sum(-1)
    union = ((masks1 + masks2) >= 1).sum(-1).clamp(min=1)

    iou = inter / union

    return iou