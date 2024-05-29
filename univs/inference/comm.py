import math
import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

import pycocotools.mask as mask_util


def generate_temporal_weights(num_frames, weights=None, enable_softmax=False, scaler=5.):
    """
    num_frames: the number of frames in the temporal dimension 
       weights: ...xT, the weights of each frame, it should be a vector with the same length with the number of frames
    """
    temp_w = (torch.arange(1, num_frames+1).float() / num_frames * scaler).exp()
    if enable_softmax:
        temp_w = temp_w.softmax(-1)

    if weights is not None:
        assert weights.shape[-1] == num_frames, "Inconsistency length between weights and values"
        temp_w = temp_w.to(weights) * weights

    return temp_w / temp_w.sum(-1).unsqueeze(-1).clamp(min=1e-3)
    
def match_from_learnable_embds(tgt_embds, cur_embds, return_similarity=False, return_src_indices=False, use_norm=True, thresh=0):
    """ !! Important 
    tgt_embds: NxT_prevxC
    cur_embds: MxT_clipxC
    use_norm: Cosine similarity if Ture, else Bisoftmax (also called as quasi_track)
    """
    num_frames_prev = tgt_embds.shape[1]
    num_frames_cur = cur_embds.shape[1]

    if use_norm:
        cur_embds = cur_embds / cur_embds.norm(dim=-1)[..., None].clamp(min=1e-3)
        tgt_embds = tgt_embds / tgt_embds.norm(dim=-1)[..., None].clamp(min=1e-3)
        cos_sim = torch.einsum('nvc,mtc->nmvt', tgt_embds, cur_embds).mean(-1)
    else:
        cos_sim = torch.einsum('nvc,mtc->nmvt', tgt_embds, cur_embds).mean(-1)
        cos_sim = cos_sim / math.sqrt(tgt_embds.shape[-1])
    
    if use_norm:
        # cosine similarity
        nonblank = (tgt_embds != 0).any(-1).float()
        temp_weight = generate_temporal_weights(num_frames_prev, weights=nonblank, enable_softmax=False)
        cos_sim = (cos_sim * temp_weight.unsqueeze(1)).sum(-1)  # NM
    else:
        # bisoftmax
        cos_sim = (cos_sim.softmax(1) + cos_sim.softmax(0)).mean(-1) / 2.
        if thresh > 0:
            cos_sim[cos_sim < thresh] = 0.

    C = (1 - cos_sim).cpu()
    indices = linear_sum_assignment(C)  # target x current
    cos_sim = cos_sim[indices]
    if not return_src_indices:
        indices = indices[1]  # permutation that makes current aligns to target

    if return_similarity:
        return indices, cos_sim
    else:
        return indices

def check_consistency_with_prev_frames(
    prev_embds, cur_embds, sim_threshold=0.5, return_similarity=False, use_norm=True
):
    """
    to remove inconsistency objects with low similarity during zeroshot inference, 
    which does not need if the model has been trained on video datasets
    prev_embds: NxT_prevxC
    cur_embds:  NxT_clipxC
    """
    num_frames_prev = prev_embds.shape[1]
    num_frames_cur = cur_embds.shape[1]

    if use_norm:
        cur_embds = cur_embds / cur_embds.norm(dim=-1)[..., None].clamp(min=1e-3)
        prev_embds = prev_embds / prev_embds.norm(dim=-1)[..., None].clamp(min=1e-3)
        cos_sim = torch.einsum('nvc,ntc->nvt', prev_embds, cur_embds).mean(-1)

        nonblank = (prev_embds != 0).any(-1).float()
        temp_weight = generate_temporal_weights(num_frames_prev, weights=nonblank, enable_softmax=False)
        cos_sim = (cos_sim * temp_weight).sum(-1)
        is_consistency = cos_sim > sim_threshold
    else:
        cos_sim = torch.einsum('nc,mc->nm', prev_embds[:,-3:].mean(1), cur_embds.mean(1))
        cos_sim = 0.5 * (cos_sim.softmax(0) + cos_sim.softmax(1))
        is_consistency = cos_sim.argmax(-1) == torch.arange(len(cos_sim), device=cos_sim.device)
        cos_sim = torch.diagonal(cos_sim, 0)
        is_consistency = is_consistency | (cos_sim > 0.25)
    
    if return_similarity:
        return is_consistency, cos_sim
    else:
        return is_consistency
    
def vis_clip_instances_to_coco_json_video(batched_inputs, results_list, apply_cls_thresh=0.05, test_topk_per_video=25):
    """
    batched_inputs: A dict to store input information, output by datamapper
    results_list: A list to store predicted results frame by frame, clip by clip, even window by window
    results_list = {
            "obj_id": 0,
            "score": np.array([0.1, 0.6, ...]),
            "segmentations": segms,
            "frame_id_start": frame_id_start
        }
    """
    assert len(batched_inputs) == 1, "More than one inputs are loaded for inference!"
    
    try: 
        video_id = int(batched_inputs[0]["video_id"])
    except Exception as err:
        print("Can not convert video in to int number!")
        video_id = batched_inputs[0]["video_id"]
    video_len = int(batched_inputs[0]["video_len"])
    height = int(batched_inputs[0]["height"])
    width = int(batched_inputs[0]["width"])

    blank_rle_mask = mask_util.encode(np.zeros((height, width, 1), order="F", dtype="uint8"))[0]
    blank_rle_mask["counts"] = blank_rle_mask["counts"].decode("utf-8")

    ytvis_results = []
    ytvis_scores = []

    num_objs_above_thresh = 0
    obj_ids = set([res["obj_id"] for res in sum(results_list, [])])
    for obj_id in obj_ids:
        obj_dict = {
            "video_id": video_id,
            "obj_id": obj_id,
            "score": [],
            "segmentations": [blank_rle_mask] * video_len,
        }

        mask_quality_score = []
        for w_i, results in enumerate(results_list):
            if len(results) == 0:
                continue

            for res in results:
                if res["obj_id"] != obj_id:
                    continue
            
                if 'mask_quality_score' in res:
                    mask_quality_score.append(res['mask_quality_score'])

                # K, class scores
                obj_dict["score"].append(res["score"])
                # List with T frames, where masks have been encoded
                f_id_s = res["frame_id_start"]
                f_id_e = f_id_s + len(res["segmentations"])
                obj_dict["segmentations"][f_id_s:f_id_e] = res["segmentations"]

        assert len(obj_dict["segmentations"]) == video_len, \
            f'The video has {video_len} frames, but the prediction has {len(obj_dict["segmentations"])} frames!'

        assert len(obj_dict["score"]), "Miss category scores here!"
        scores = torch.stack(obj_dict["score"], dim=0)

        if len(mask_quality_score):
            mask_quality_score = sum(mask_quality_score) / len(mask_quality_score)
        else:
            nonblank_len = (scores.sum(-1) > 0).sum(0)
            mask_quality_score = (nonblank_len / video_len).clamp(min=0.1)

        scores = calculate_mask_temporal_consistency_scores(scores)
        scores = scores.sum(0) / (scores.sum(-1) > 0).sum(0).clamp(min=1)
        classes = np.arange(len(scores))

        segm = obj_dict["segmentations"]
        for c in classes:
            if float(scores[c]) < 0.1 * apply_cls_thresh:
                continue
            
            s = float(scores[c]) * float(mask_quality_score)
            l = int(c)
            ytvis_results.append({
                "video_id": video_id,
                "score": s,
                "category_id": l,
                "segmentations": segm,
                "height": height,
                "width": width
            })
            ytvis_scores.append(s)
            if scores[c] > apply_cls_thresh:
                num_objs_above_thresh += 1

    if len(ytvis_scores):
        ytvis_scores.sort()
        num_topk = max(int(num_objs_above_thresh*1.5), test_topk_per_video)
        topk_score = ytvis_scores[::-1][min(num_topk, len(ytvis_scores)-1)]
        ytvis_results = [r for r in ytvis_results if r['score'] >= topk_score]

    return ytvis_results

def calculate_mask_temporal_consistency_scores(scores):
    nonblank = scores.sum(-1) > 0

    dt = 1
    for t in range(len(nonblank)):
        s_t = max(0, t-dt)
        e_t = min(len(nonblank), t+dt)
        w = nonblank[t] * nonblank[s_t:e_t].sum() / max(e_t-s_t, 1)
        scores[t] *= w 
    
    return scores
