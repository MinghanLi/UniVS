import cv2
import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode, GenericMask, Visualizer, _create_text_labels

from mdqe.data.dataset_mapper import build_augmentation

from visualizer import ClipVisualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = ClipPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions, images = self.predictor(image)

        max_h, max_w = 0, 0
        for image in images:
            h, w = image.shape[:2]
            max_h = h if max_h < h else max_h
            max_w = w if max_w < w else max_w

        concat_image = np.zeros((max_h, max_w * len(images), 3))
        for i, image in enumerate(images):
            h, w = image.shape[:2]
            concat_image[:h, max_w*i:max_w*i+w, :] = image

        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        concat_image = concat_image[:, :, ::-1]
        visualizer = ClipVisualizer(concat_image, len(images), max_h, max_w, metadata=self.metadata, instance_mode=self.instance_mode)

        instances = [p["instances"].to(self.cpu_device) for p in predictions]
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


class ClipPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        self.aug = T.AugmentationList(
            [T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)] + 
            build_augmentation(cfg, True)
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            inputs = []
            images = []
            for _ in range(self.frame_num):
                aug_input = T.AugInput(original_image)
                transforms = self.aug(aug_input)
                image = aug_input.image

                #image = self.aug.get_transform(original_image).apply_image(original_image)
                images.append(image[:, :, ::-1])
                height, width = image.shape[:2]
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]

                inputs.append({"image": image, "height": height, "width": width})
            predictions = self.model(inputs)
            return predictions, images
