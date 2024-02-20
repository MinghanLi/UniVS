# https://github.com/MinghanLi/MDQE_CVPR2023/blob/main/mdqe/tracking/OverTracker.py
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Any, Dict, List, Tuple, Union

from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom


class FastOverTracker_DET:
    """
    https://github.com/MinghanLi/MDQE_CVPR2023/blob/main/mdqe/tracking/OverTracker.py
    This structure is to support instance tracking (long) clip by (long) clip, which is memory friendly for long videos.
     We only store the instance masks of a long clip, instead of all instance masks in the whole video.
    """

    @configurable
    def __init__(
            self,
            num_classes,
            num_frames,
            num_frames_window_track,
            clip_stride,
            embed_dim,
            apply_cls_thres,
            mask_classification
    ):
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.window_frames = num_frames_window_track
        self.clip_stride = clip_stride
        self.embed_dim = embed_dim
        self.apply_cls_thres = apply_cls_thres
        self.mask_classification = mask_classification

        self.mem_length = num_frames_window_track + num_frames + 1
        self.num_clips = num_frames_window_track // self.clip_stride + num_frames 

        # cost matrix params
        self.siou_match_threshold = 0.05
        self.ctt_match_threshold = 0.6
        self.beta_siou = 1
        self.beta_ctt = 1

        self.weighted_manner = True
        self.num_clip_mem_long = 10 // self.clip_stride
        self.weights_mem = torch.exp(torch.arange(self.num_clip_mem_long) * 0.25)

        self.num_max_inst = None
        self.saved_frame_idx = range(0, self.mem_length)
        self.saved_inst_id = None

        # init_memory
        self.image_size = None
        self.device = 'cpu'
        self.num_inst = 0
        self.num_inst_prev_windows = 0
        self.num_clip = 0
        self.saved_idx_set = set()

    @classmethod
    def from_config(cls, cfg):
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        return {
            # inference
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_frames_window_track": cfg.MODEL.BoxVIS.TEST.NUM_FRAMES_WINDOW,
            "clip_stride":cfg.MODEL.BoxVIS.TEST.CLIP_STRIDE,
            "embed_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "apply_cls_thres": cfg.MODEL.BoxVIS.TEST.APPLY_CLS_THRES,
            "mask_classification": True,
        }

    def init_memory(self, num_insts=0, is_first=False, image_size=None, device=None, num_classes=None):
        if num_insts == 0 or is_first:
            self.num_max_inst = 1
        else:
            self.num_max_inst = int(1.5 * self.num_inst) if self.num_inst < 50 else int(1.2 * self.num_inst)
        if num_classes is not None:
            self.num_classes = num_classes

        if is_first:
            assert image_size and device
            self.image_size = image_size
            self.device = device
            self.num_inst = 0
            self.num_inst_prev_windows = 0
            self.num_clip = 0
            self.saved_idx_set = set()
            self.saved_inst_id = torch.arange(self.num_max_inst, device=device)
        else:
            self.num_clip = 1
            self.num_inst = num_insts
            self.saved_idx_set = set(range(self.num_frames - 1))

        self.saved_logits = torch.zeros((self.num_clips, self.num_max_inst, self.mem_length, *self.image_size),
                                        dtype=torch.float, device=self.device)

        self.saved_valid = torch.zeros((self.num_clips, self.num_max_inst, self.mem_length),
                                       dtype=torch.bool, device=self.device)
        
        self.saved_cls = torch.zeros((self.num_clips, self.num_max_inst, self.num_classes),
                                        dtype=torch.float, device=self.device)
        self.saved_query_embeds = torch.zeros((self.num_clips, self.num_max_inst, self.embed_dim),
                                              dtype=torch.float, device=self.device)

        self.saved_untracked_frames_mem = torch.zeros(self.num_max_inst,
                                                      dtype=torch.float, device=self.device)
        self.saved_query_embeds_mem = torch.zeros((self.num_max_inst, self.embed_dim),
                                                  dtype=torch.float, device=self.device)

    def _expand_memory(self, num_expand_inst):
        expand_logits = torch.zeros((self.num_clips, num_expand_inst, self.mem_length, *self.image_size),
                                    dtype=torch.float, device=self.device)
        expand_valid = torch.zeros((self.num_clips, num_expand_inst, self.mem_length),
                                   dtype=torch.bool, device=self.device)
        expand_cls = torch.zeros((self.num_clips, num_expand_inst, self.num_classes),
                                 dtype=torch.float, device=self.device)
        expand_query_embeds = torch.zeros((self.num_clips, num_expand_inst, self.embed_dim),
                                          dtype=torch.float, device=self.device)
        expand_untracked_frames_mem = torch.zeros(num_expand_inst,
                                                  dtype=torch.float, device=self.device)
        expand_query_embeds_mem = torch.zeros((num_expand_inst, self.embed_dim),
                                              dtype=torch.float, device=self.device)
        
        self.saved_logits = torch.cat([self.saved_logits, expand_logits], dim=1)
        self.saved_valid = torch.cat([self.saved_valid, expand_valid], dim=1)
        self.saved_cls = torch.cat([self.saved_cls, expand_cls], dim=1)
        self.saved_query_embeds = torch.cat([self.saved_query_embeds, expand_query_embeds], dim=1)
        self.saved_untracked_frames_mem = torch.cat([self.saved_untracked_frames_mem, expand_untracked_frames_mem])
        self.saved_query_embeds_mem = torch.cat([self.saved_query_embeds_mem, expand_query_embeds_mem])

        max_inst_id = max(self.saved_inst_id) + 1
        self.saved_inst_id = torch.cat([
            self.saved_inst_id, max_inst_id + torch.arange(num_expand_inst, device=self.device)
        ])
        self.num_max_inst += num_expand_inst

    def _update_memory(self, r_idx=None, c_idx=None, input_clip=None):
        start_idx = min(input_clip.frame_idx)
        end_idx = max(input_clip.frame_idx)

        if len(r_idx) and max(r_idx) >= self.num_max_inst:
            num_expand_inst = len([1 for idx in r_idx if idx >= self.num_max_inst])
            self._expand_memory(num_expand_inst)

        assert len(r_idx) == len(c_idx), f"Length mismatch {len(r_idx)} and {len(c_idx)}"
        self.saved_logits[self.num_clip, r_idx, start_idx:end_idx + 1] = \
            input_clip.mask_logits[c_idx].float()
        self.saved_valid[self.num_clip, r_idx, start_idx:end_idx + 1] = True
        self.saved_cls[self.num_clip, r_idx] = input_clip.cls_probs[c_idx].float()
        self.saved_query_embeds[self.num_clip, r_idx] = input_clip.query_embeds[c_idx].float()

        # update mem pool
        self.saved_untracked_frames_mem += 1
        self.saved_untracked_frames_mem[r_idx] = 0
        if self.num_clip > 0 and self.weighted_manner:
            start_clip_idx = max(self.num_clip - 1, 0)
            query_embed_mem = self.saved_query_embeds[start_clip_idx:self.num_clip + 1][:, r_idx]  # CxNxE
            w_mem = self.weights_mem[:query_embed_mem.shape[0]].reshape(-1, 1, 1).to(self.device)
            valid_mem = (query_embed_mem != 0).any(dim=-1)[..., None]  # CxNx1
            query_embed_mem_w = (query_embed_mem * w_mem).sum(dim=0)
            valid_mem_w = (valid_mem * w_mem).sum(dim=0).clamp(min=1)
            self.saved_query_embeds_mem[r_idx] = query_embed_mem_w / valid_mem_w  # NxE
        else:
            self.saved_query_embeds_mem[r_idx] = input_clip.query_embeds[c_idx].float()

    def _get_siou(self, saved_masks, input_masks):
        # input_masks : N_i, T, H, W
        # saved_masks : N_s, T, H, W
        # downsample masks for memory friendly (crowded object)
        H, W = input_masks.shape[-2:]
        if saved_masks.shape[0] >= 20 or input_masks.shape[0] >= 20:
            input_masks = retry_if_cuda_oom(F.interpolate)(
                input_masks,
                size=(int(H/2), int(W/2)),
                mode="bilinear",
                align_corners=False,
            )  # N_i, T, H, W
            saved_masks = retry_if_cuda_oom(F.interpolate)(
                saved_masks,
                size=(int(H/2), int(W/2)),
                mode="bilinear",
                align_corners=False,
            )  # N_s, T, H, W

        input_masks = input_masks.flatten(1).gt(0.5).float()
        saved_masks = saved_masks.flatten(1).gt(0.5).float()

        if saved_masks.shape[0] >= 50:
            # convert masks into CPT for memory friendly (crowded object)
            input_masks = input_masks.cpu()
            saved_masks = saved_masks.cpu()

        input_masks = input_masks.unsqueeze(0)  # 1, N_i, THW
        saved_masks = saved_masks.unsqueeze(1)  # N_s, 1,  THW

        # N_s, N_i, THW
        numerator = saved_masks * input_masks
        denominator = saved_masks + input_masks - numerator
        siou = numerator.sum(-1) / denominator.sum(-1).clamp(min=1)  # N_s, N_i

        return siou.to(self.device)

    def update(self, input_clip):

        if self.num_inst == 0:
            matched_ID = matched_idx = list(range(len(input_clip.scores)))
            self.num_inst += len(input_clip.scores)
        else:

            # 1. Compute the score_mem of bi-softmax similarity: long matching + short matching
            query_embed_mem = self.saved_query_embeds_mem[:self.num_inst]
            still_appeared_long = (
                self.saved_untracked_frames_mem[:self.num_inst] < self.num_clip_mem_long
            ).nonzero().reshape(-1).tolist()

            scores_mem = torch.zeros(self.num_inst, input_clip.query_embeds.shape[0], device=self.device)
            scores_mem[still_appeared_long] = get_ctt_similarity(query_embed_mem[still_appeared_long],
                                                                 input_clip.query_embeds)

            # 2. Compute the mask iou on overlapping frames
            inter_input_idx, inter_saved_idx = [], []
            for o_i, f_i in enumerate(input_clip.frame_idx):
                if f_i in self.saved_idx_set:
                    inter_input_idx.append(o_i)
                    inter_saved_idx.append(self.saved_frame_idx.index(f_i))

            siou_scores = torch.zeros(query_embed_mem.shape[0], input_clip.query_embeds.shape[0], device=self.device)
            if len(inter_saved_idx) > 0:
                if self.beta_siou > 0:
                    i_masks = input_clip.mask_logits[:, inter_input_idx].float()
                    s_masks = self.saved_logits[:self.num_clip, :self.num_inst, inter_saved_idx]
                    s_valid = self.saved_valid[:self.num_clip, :self.num_inst].any(dim=-1).to(s_masks.device)
                    s_masks = (s_masks.sum(0) / s_valid.sum(0).clamp(min=1).reshape(-1, 1, 1, 1))
                    siou_scores = self._get_siou(s_masks.sigmoid(), i_masks.sigmoid())  # N_s, N_i

            # 3. Combine score matrix
            scores = self.beta_ctt * scores_mem + self.beta_siou * siou_scores
            match_threshold = self.beta_ctt * self.ctt_match_threshold + \
                              self.beta_siou * self.siou_match_threshold
            above_thres = scores > match_threshold
            scores = scores * above_thres.float()

            row_idx, col_idx = linear_sum_assignment(scores.cpu(), maximize=True)

            matched_ID, matched_idx = [], []
            for is_above, r, c in zip(above_thres[row_idx, col_idx], row_idx, col_idx):
                if not is_above:
                    continue
                matched_idx.append(c)
                matched_ID.append(r)
                siou_scores[r, c] = -1
                scores_mem[r, c] = 0

            # 4. Remove repeatedly-detected objects with high mask IoU
            unmatched_idx = [int(idx) for idx in range(len(input_clip.scores)) if idx not in matched_idx]
            repeated_idx, repeated_siou = [], []
            for idx in unmatched_idx:
                max_matched_siou = siou_scores[:, idx].max(dim=0)[0]
                max_matched_ctt = scores_mem[:, idx].max(dim=0)[0]
                if max_matched_siou > 0.4 or max_matched_ctt > self.ctt_match_threshold:
                    repeated_idx.append(idx)
                    repeated_siou.append([max_matched_siou, max_matched_ctt])

            unmatched_idx = [int(idx) for idx in range(len(input_clip.scores))
                             if idx not in matched_idx + repeated_idx and input_clip.cls_probs[idx].max() > self.apply_cls_thres]

            new_assign_ID = list(range(self.num_inst, self.num_inst + len(unmatched_idx)))
            matched_ID = matched_ID + new_assign_ID
            matched_idx = matched_idx + unmatched_idx
            self.num_inst += len(new_assign_ID)

        # Update memory
        self._update_memory(matched_ID, matched_idx, input_clip)

        # Update status
        self.saved_idx_set.update(set(input_clip.frame_idx))
        self.num_clip += 1

    def get_query_embds(self):
        return self.saved_query_embeds_mem[:self.num_inst]

    def get_result(self, is_last_window=False):
        mask_logits = self.saved_logits[:self.num_clip, :self.num_inst]  # CxNxTxHxW
        valid = self.saved_valid[:self.num_clip, :self.num_inst]  # CxNxT

        mask_logits = mask_logits.sum(dim=0) / valid.sum(dim=0).clamp(min=1)[..., None, None].to(mask_logits.device)  # NxTxHxW
        len_frames = self.window_frames if not is_last_window else max(self.saved_idx_set) + 1
        out_masks = mask_logits[:, :len_frames]  # NxTxHxW

        cls_probs = self.saved_cls[:self.num_clip, :self.num_inst]  # CxNxK
        valid_clip = valid.any(dim=-1)[..., None]  # CxNx1
        out_cls = (cls_probs * valid_clip).sum(dim=0) / valid_clip.sum(dim=0).clamp(min=1)  # NxK

        # update query embedding
        query_embeds_mem = self.saved_query_embeds_mem[:self.num_inst]
        untracked_frames_mem = self.saved_untracked_frames_mem[:self.num_inst]
        out_inst_id = self.saved_inst_id[:self.num_inst]

        if not is_last_window:
            # newly appeared instance in the cur window
            valid_track = untracked_frames_mem < self.num_clip_mem_long  # N
            valid_cls = out_cls.max(dim=-1)[0] > self.apply_cls_thres  # N
            valid_inst_cur = valid_cls | valid_track
            num_insts_cur = int(valid_inst_cur.sum())

            # update memory pool for the next window
            self.init_memory(num_insts_cur)
            if self.num_frames > 1:
                self.saved_logits[0, :self.num_inst, :self.mem_length - self.window_frames] = \
                    mask_logits[:, self.window_frames:][valid_inst_cur]
                self.saved_valid[0, :self.num_inst, :self.mem_length - self.window_frames] = \
                    valid[-self.num_frames+1:, :, self.window_frames:].any(dim=0)[valid_inst_cur]
                self.saved_query_embeds[0, :self.num_inst] = query_embeds_mem[valid_inst_cur]
                self.saved_cls[0, :self.num_inst] = out_cls[valid_inst_cur]

            # update instance embeddings in memory pool
            self.saved_query_embeds_mem[:self.num_inst] = query_embeds_mem[valid_inst_cur]
            self.saved_untracked_frames_mem[:self.num_inst] = untracked_frames_mem[valid_inst_cur]

            # update instance IDs in memory pool
            saved_inst_id = out_inst_id[valid_inst_cur]
            num_inst_newly = (saved_inst_id >= self.num_inst_prev_windows).sum()
            newly_inst_id = self.num_inst_prev_windows + torch.arange(num_inst_newly, device=self.device)
            if num_inst_newly > 0:
                saved_inst_id[-num_inst_newly:] = newly_inst_id

            # remove objects with low confident scores in output
            valid_inst_prev = (out_inst_id < self.num_inst_prev_windows).to(self.device)
            valid_inst_out = valid_inst_prev | valid_inst_cur
            out_cls = out_cls[valid_inst_out]
            out_masks = out_masks[valid_inst_out]
            out_inst_id = out_inst_id[valid_inst_out]
            if num_inst_newly > 0:
                out_inst_id[-num_inst_newly:] = newly_inst_id

            self.num_inst_prev_windows += num_inst_newly
            self.saved_inst_id = torch.cat([
                saved_inst_id,
                torch.arange(self.num_max_inst, device=self.device) + self.num_inst_prev_windows
            ])[:self.num_max_inst]

        return {"pred_masks": out_masks, "pred_cls_scores": out_cls, "obj_ids": out_inst_id}


def get_ctt_similarity(saved_query_embeds, input_query_embeds):
    # input_query_embeds: N_i, E
    # saved_query_embeds: N_s, E
    feats = torch.einsum('nd,md->nm', saved_query_embeds, input_query_embeds)  # N_s, N_i
    Ns, Ni = feats.shape
    Ws = 1 if Ns > 1 else 0
    Wi = 1 if Ni > 1 else 0
    d2t_scores = feats.softmax(dim=0)
    t2d_scores = feats.softmax(dim=1)
    if Ns == 1 and Ni == 1:
        scores = 0.5 * (d2t_scores + t2d_scores)
    else:
        scores = (Ws * d2t_scores + Wi * t2d_scores) / max(Ws + Wi, 1)

    return scores


def get_cos_similarity(saved_query_embeds, input_query_embeds):
    return torch.einsum('nd,md->nm', F.normalize(saved_query_embeds),
                        F.normalize(input_query_embeds))  # N_s, N_i


class Clips(Instances):
    def __init__(self, image_size: Tuple[int, int], frame_idx: List[int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        super().__init__(image_size, **kwargs)
        self._image_size = image_size
        self._frame_idx = frame_idx
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def frame_idx(self) -> List:
        """
        Returns:
            list: the index of frames in the video clip
        """
        return self._frame_idx


class InterClipCombiner_SOT:
    """
    sot or grounding datasets to associate objects across clips
    """
    @configurable
    def __init__(self, num_frames_clip):
        self.num_insts = -1
        self.num_frames_clip = num_frames_clip
        assert num_frames_clip == 2

        self.memory_outs = dict()
        self.num_frames_memory = self.num_frames_clip
        self.num_clips_memory = self.num_frames_clip

        self.frame_idxs = range(self.num_frames_memory)
        self.valid = None

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_frames_clip": cfg.MODEL.VideoSAM.TEST.NUM_FRAMES,
        }

    def _init_memory(self, frame_idx, outs, is_first=False):
        self.num_insts = outs["pred_masks"].shape[0]
        shape_mem = (self.num_clips_memory, self.num_insts, self.num_frames_memory)

        if is_first or len(self.memory_outs) == 0:
            assert outs is not None
            device = outs["pred_masks"].device

            for k, v in outs.items():
                if k.startswith("ids"):
                    self.memory_outs[k] = v.reshape(-1).tolist()
                else:
                    assert len(v) == len(outs["ids_occur"]), "Mismatch number of pred and gt objects"
                    v_shape_mem = (*shape_mem, *v.shape[2:])
                    self.memory_outs[k] = torch.zeros(v_shape_mem, dtype=v.dtype, device=device)
            self.valid = torch.zeros(shape_mem, device=device).bool()
            self.frame_idxs = range(frame_idx, frame_idx + self.num_frames_memory)

        else:
            device = self.valid.device
            _ids_occur_mem = [outs["ids_occur"].tolist().index(_id)
                              for _id in self.memory_outs["ids_occur"]]

            for k, v in self.memory_outs.items():
                if k.startswith("ids"):
                    self.memory_outs[k] = outs[k].reshape(-1).tolist()
                else:
                    v_shape_mem = (*shape_mem, *v.shape[3:])
                    old_v = copy.deepcopy(v)
                    self.memory_outs[k] = torch.zeros(v_shape_mem, dtype=v.dtype, device=device)
                    self.memory_outs[k][:-1, _ids_occur_mem, :-1] = old_v[1:, :, 1:]

            old_valid = copy.deepcopy(self.valid)
            self.valid = torch.zeros(shape_mem, device=device).bool()
            self.valid[:-1, _ids_occur_mem, :-1] = copy.deepcopy(old_valid[1:, :, 1:])
            self.frame_idxs = range(frame_idx, frame_idx + self.num_frames_memory)

    def update_memory(self, frame_idx, cur_clip_out, is_first=False, is_last=False):
        self._init_memory(frame_idx, cur_clip_out, is_first=is_first)

        assert self.frame_idxs[0] == frame_idx, \
            "dismatch frame index: {:0>3d} and {:0>3d}".format(self.frame_idxs[0], frame_idx)

        # only update masks with high reid score
        keep = cur_clip_out['pred_cls_scores'].flatten(1).min(1)[0] > 0.25
        for k, v in cur_clip_out.items():
            if not k.startswith("ids"):
                self.memory_outs[k][-1, keep] = v[keep]
        self.valid[-1, keep] = 1

        num_valid_frames = 1 if not is_last else self.num_frames_clip

        # output for the current frame
        valid = self.valid[:, :, :num_valid_frames].sum(0).clamp(min=1)

        outs = dict()
        for k, v in self.memory_outs.items():
            if k.startswith("ids"):
                outs[k] = v
            else:
                _valid = valid.reshape(-1, num_valid_frames, *([1] * (v.dim() - 3)))
                if k.startswith("pred_masks"):
                    outs[k] = v[:, :, :num_valid_frames].sum(0) / _valid
                else:
                    outs[k] = (v[:, :, :num_valid_frames].sum(0) / _valid).mean(1)

        if is_last:
            self.memory_outs = dict()

        return outs

    def get_next_frame_predictions(self):
        outs = dict()
        for k, v in self.memory_outs.items():
            outs[k] = v if k.startswith("ids") else v[-1, :, 1]

        return outs



