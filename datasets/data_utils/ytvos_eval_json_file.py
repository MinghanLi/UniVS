import json
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval


def _evaluate_predictions_on_ytvis(vis_gt, vis_results, iou_type="segm"):
    vis_dt = vis_gt.loadRes(vis_results)
    vis_eval = YTVOSeval(vis_gt, vis_dt, iouType=iou_type)

    vis_eval.evaluate()
    vis_eval.accumulate()
    vis_eval.summarize()

    return vis_eval.stats


json_file = '/data1/lmh/code/MDQE/datasets/ovis/valid_sub.json'
ytvis_results = '/data1/lmh/code/MDQE/output/coco/mdqe_r50_coco+ovis_f5c2_360_re/inference/results.json'

_ytvis_api = YTVOS(json_file)

ytvis_eval = _evaluate_predictions_on_ytvis(_ytvis_api, ytvis_results)
