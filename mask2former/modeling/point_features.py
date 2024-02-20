# Copyright (c) Facebook, Inc. and its affiliates.
import math
import torch
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.structures import BitMasks, Boxes
from detectron2.projects.point_rend.point_features import (
    generate_regular_grid_point_coords,
    get_point_coords_wrt_image,
    point_sample
)


"""
Shape shorthand in this module:

    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""


def get_uncertain_point_coords_on_grid_boxvis(
    boxes_coords, coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        boxes_coords (Tensor): A tensor of shape (N, 4) for bounding boxes
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]

    # sample points along with grid in bounding boxes
    side_size = int(math.sqrt(num_points) // 2)
    point_coords_grid_inbox = generate_regular_grid_point_coords(num_boxes, side_size, coarse_logits.device)
    point_coords_grid_inbox = get_point_coords_wrt_image(boxes_coords, point_coords_grid_inbox)

    num_points_outbox = (num_points-side_size**2)
    num_sampled = int(num_points_outbox * oversample_ratio)
    point_coords_random_outbox = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords_random_outbox, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points_outbox)
    num_random_points = num_points_outbox - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords_random_outbox = point_coords_random_outbox.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    point_coords = torch.cat([point_coords_grid_inbox, point_coords_random_outbox], dim=1)
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def get_uncertain_point_coords_inbox(
    boxes_coords, coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        boxes_coords (Tensor): A tensor of shape (N, 4) for bounding boxes
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]

    # sample points along with grid in bounding boxes
    side_size = int(math.sqrt(num_points * oversample_ratio))
    num_sampled = side_size ** 2

    point_coords_grid_inbox = generate_regular_grid_point_coords(num_boxes, side_size, coarse_logits.device)
    point_coords_grid_inbox = get_point_coords_wrt_image(boxes_coords, point_coords_grid_inbox)
    point_logits = point_sample(coarse_logits, point_coords_grid_inbox, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords_grid_inbox.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )

    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords
