import numpy as np
import torch

import matplotlib.colors as mplc

from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask, _create_text_labels, _OFF_WHITE, _PanopticPrediction
from detectron2.utils.colormap import _COLORS


class VisualizerFrame(Visualizer):
    def __init__(
        self, 
        img_rgb, 
        metadata=None, 
        scale=1.0, 
        instance_mode=ColorMode.IMAGE, 
        class_names=None,
    ):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.class_names = class_names  # open-voc category names

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        class_names = self.metadata.get("thing_classes", None) if self.class_names is None else self.class_names
        labels = _create_text_labels(classes, scores, class_names)

        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        colors = produce_colors(len(scores), rgb=True, maximum=1)
        alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
    
    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.6):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = self.metadata.stuff_classes[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output
    
    def draw_panoptic_seg(self, panoptic_seg, segments_info, area_threshold=None, alpha=0.6):
        """
        Draw panoptic prediction annotations or results.

        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict] or None): Describe each segment in `panoptic_seg`.
                If it is a ``list[dict]``, each dict contains keys "id", "category_id".
                If None, category id of each pixel is computed by
                ``pixel // metadata.label_divisor``.
            area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

        Returns:
            output (VisImage): image object with visualizations.
        """
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))

        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None
            
            text = self.metadata.stuff_classes[category_idx]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        # draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]
        obj_ids = [x["id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        labels = _create_text_labels(
            category_ids, scores, self.metadata.thing_classes, [x.get("iscrowd", 0) for x in sinfo]
        )

        try:
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]], obj_id) for (c, obj_id) in zip(category_ids, obj_ids)
            ]
        except AttributeError:
            colors = None
        
        self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        return self.output
    
    def _jitter(self, color, obj_id=None):
        """
        Randomly modifies given color to produce a slightly different color than the color given.

        Args:
            color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
                picked. The values in the list are in the [0.0, 1.0] range.

        Returns:
            jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
                color after being jittered. The values in the list are in the [0.0, 1.0] range.
        """
        color = mplc.to_rgb(color)
        if obj_id is None:
            vec = np.random.rand(3)
        else:
            vec = np.array([obj_id, obj_id+1, obj_id+2])
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)


def produce_colors(N, rgb=False, maximum=255):
    """
    Args:
        N (int): number of unique colors needed
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a list of random_color
    """
    indices = range(N)
    ret = [_COLORS[i % len(_COLORS)] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret