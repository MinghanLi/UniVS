import torch
import numpy as np

from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import ColorMode, GenericMask, Visualizer, _create_text_labels


_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)


class ClipVisualizer(Visualizer):

    def __init__(self, img_rgb, num_frames, max_h, max_w, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): image metadata.
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        super().__init__(
            img_rgb, metadata=metadata, scale=scale, instance_mode=instance_mode
        )
        self.num_frames = num_frames
        self.max_h = max_h
        self.max_w = max_w

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
        boxes = [p.pred_boxes if p.has("pred_boxes") else None for p in predictions]
        scores = [p.scores if p.has("scores") else None for p in predictions]
        classes = [p.pred_classes if p.has("pred_classes") else None for p in predictions]
        labels = _create_text_labels(classes[0], scores[0], self.metadata.get("thing_classes", None))

        if predictions[0].has("pred_masks"):
            h = self.output.height
            w = self.output.width // self.num_frames
            masks = []
            for i, p in enumerate(predictions):
                n, p_h, p_w = p.pred_masks.shape
                p_masks = np.zeros((n, h, self.output.width), dtype=np.bool)
                p_masks[:, :p_h, i*w:i*w+p_w] = np.asarray(p.pred_masks)
                masks.append([GenericMask(x, self.output.height, self.output.width) for x in p_masks])
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        assigned_colors=None,
        alpha=0.5
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = [self._convert_boxes(b) for b in boxes]
            num_instances = len(boxes[0])
        if masks is not None:
            masks = [self._convert_masks(m) for m in masks]
            if num_instances:
                assert len(masks[0]) == num_instances
            else:
                num_instances = len(masks[0])
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = sum([np.prod(b[:, 2:] - b[:, :2], axis=1) for b in boxes])
        elif masks is not None:
            areas = np.asarray(sum([[x.area() for x in m] for m in masks]))

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = [b[sorted_idxs] for b in boxes] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [[m[idx] for idx in sorted_idxs] for m in masks] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        for frame_idx in range(self.num_frames):
            cur_offset = self.max_w * frame_idx
            cur_boxes, cur_masks = None, None
            if boxes is not None:
                cur_boxes = boxes[frame_idx] + np.array([cur_offset, 0, cur_offset, 0])
            if masks is not None:
                cur_masks = masks[frame_idx]

            for i in range(num_instances):
                color = assigned_colors[i]

                if cur_boxes is not None:
                    self.draw_box(cur_boxes[i], edge_color=color)

                if masks is not None:
                    for segment in cur_masks[i].polygons:
                        self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

                if labels is not None:
                    # first get a box
                    if cur_boxes is not None:
                        x0, y0, x1, y1 = cur_boxes[i]
                        text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                        horiz_align = "left"
                    elif masks is not None:
                        # skip small mask without polygon
                        if len(masks[i].polygons) == 0:
                            continue

                        x0, y0, x1, y1 = masks[i].bbox()

                        # draw text in the center (defined by median) when box is not drawn
                        # median is less sensitive to outliers.
                        text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                        horiz_align = "center"
                    else:
                        continue  # drawing the box confidence for keypoints isn't very useful.
                    # for small objects, draw text at the side to avoid occlusion
                    instance_area = (y1 - y0) * (x1 - x0)
                    if (
                        instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                        or y1 - y0 < 40 * self.output.scale
                    ):
                        if y1 >= self.output.height - 5:
                            text_pos = (x1, y0)
                        else:
                            text_pos = (x0, y1)

                    height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                    lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                    font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                    )
                    self.draw_text(
                        labels[i],
                        text_pos,
                        color=lighter_color,
                        horizontal_alignment=horiz_align,
                        font_size=font_size,
                    )

        return self.output
