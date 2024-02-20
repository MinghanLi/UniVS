# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.file_io import PathManager

from predictor import VisualizationDemo
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from boxvis import add_boxvis_config


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_boxvis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output", required=True, help="A file or directory to save output visualizations."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint pth",
    )
    parser.add_argument(
        "--save-frames",
        default=False,
        help="Save frame level image outputs.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        
        if not os.path.isdir(args.output):
            PathManager.mkdirs(args.output)

        for vid_path in tqdm.tqdm(args.input, disable=not args.output):
            vid_file = vid_path.split("/")[-1]
            out_vid_path = os.path.join(args.output, vid_file)
            if args.save_frames and not os.path.isdir(out_vid_path):
                PathManager.mkdirs(out_vid_path)

            vid_frame_paths = sorted(PathManager.ls(vid_path))
            vid_frames = []
            for img_file in vid_frame_paths:
                img_path = os.path.join(vid_path, img_file)
                # use PIL, to be consistent with evaluation
                img = read_image(img_path, format="BGR")
                vid_frames.append(img)

            start_time = time.time()
            predictions, visualized_output = demo.run_on_video(vid_frames)
            logger.info(
                "{}: detected {} instances per frame in {:.2f}s".format(
                    vid_path, len(predictions["pred_scores"]), time.time() - start_time
                )
            )

            if args.save_frames:
                for img_file, _vis_output in zip(vid_frame_paths, visualized_output):
                    out_filename = os.path.join(out_vid_path, img_file)
                    _vis_output.save(out_filename)

            H, W = visualized_output[0].height, visualized_output[0].width

            cap = cv2.VideoCapture(-1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_vid_path + ".mp4" , fourcc, 10.0, (W, H), True)
            for _vis_output in visualized_output:
                frame = _vis_output.get_image()[:, :, ::-1]
                out.write(frame)
            cap.release()
            out.release()
