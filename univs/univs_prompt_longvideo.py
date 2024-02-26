import copy
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
# support color space pytorch, https://kornia.readthedocs.io/en/latest/_modules/kornia/color/lab.html#rgb_to_lab
from kornia import color
from scipy.optimize import linear_sum_assignment

import math
import random
import numpy as np
import pycocotools.mask as mask_util

from torchvision.ops.boxes import batched_nms, box_area

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.utils.box_ops import box_xyxy_to_cxcywh
from univs import (
    VideoSetCriterionUni, 
    VideoHungarianMatcherUni,
    BoxVISTeacherSetPseudoMask,
    TextPromptEncoder,
    build_clip_language_encoder,
    Clips, 
    MDQE_OverTrackerEfficient,
    InferenceImageGenericSegmentation,
    InferenceVideoVIS,
    InferenceVideoVISFast,
    InferenceVideoVPS,
    InferenceVideoVOS,
    InferenceVideoEntity,
    )
from univs.utils.comm import convert_mask_to_box, calculate_mask_quality_scores

from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info
from .prepare_targets import PrepareTargets


def contrastive_loss(inputs, targets, pos_topk=5, topk=30):
    if inputs.nelement() == 0:
        return inputs[:0].sum().detach()

    inputs = inputs.flatten(1)    # N, K
    targets = targets.flatten(1)  # N, K
    N, N_neg = inputs.shape
    neg_topk = min(max(N * pos_topk, topk), 50)
    
    pos_inputs = []
    for i, t in enumerate(targets):
        pos_indices = random.choices(torch.nonzero(t).reshape(-1).tolist(), k=pos_topk)
        pos_inputs.append(inputs[i, pos_indices])
    pos_inputs = torch.stack(pos_inputs)  # N K_pos
    
    neg_indices = torch.randperm(N_neg)[:neg_topk]
    inputs = inputs[:, neg_indices]       # N K_neg
    targets = targets[:, neg_indices]     # N K_neg
    negpos_inputs = (inputs[:, :, None] - pos_inputs[:, None]) * torch.logical_not(targets)[:, :, None]  # N K_neg K_pos 
    negpos_inputs = negpos_inputs.clamp(max=10.).exp() * torch.logical_not(targets)[:, :, None]  # N K_neg K_pos
    
    # loss = torch.logsumexp(inputs_negpos, dim=-1)
    loss = (1 + torch.sum(negpos_inputs.flatten(1), dim=-1)).log()
    loss = loss.sum() / max(len(loss), 1)

    return loss

def contrastive_aux_loss(inputs, targets, pos_topk=3, topk=30):
    if inputs.nelement() == 0:
        return inputs[:0].sum().detach()
    
    # assert inputs.min() >= -1 and inputs.max() <= 1
    inputs = inputs.flatten(1)    # N, K
    targets = targets.flatten(1)  # N, K
    N, N_neg = inputs.shape
    neg_topk = min(max(N * pos_topk, topk), 50)

    neg_indices = torch.randperm(N_neg)[:neg_topk]
    inputs = inputs[:, neg_indices].clamp(min=0)  # N K_neg
    targets = targets[:, neg_indices]  # N K_neg

    return F.smooth_l1_loss(inputs, targets, reduction='sum') / max(len(inputs), 1)


@META_ARCH_REGISTRY.register()
class UniVS_Prompt_LongVideo(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        inference_img_generic_seg: nn.Module,
        inference_video_vis: nn.Module,
        inference_video_vis_fast: nn.Module,
        inference_video_vps: nn.Module,
        inference_video_vos: nn.Module,
        inference_video_entity: nn.Module,
        text_prompt_encoder,
        prepare_targets, 
        hidden_dim: int,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        stability_score_thresh: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        num_frames: int,
        num_frames_window: int,
        num_frames_video: int,
        num_classes: int,
        # boxvis
        gen_pseudo_mask: nn.Module,
        boxvis_enabled: bool,
        boxvis_ema_enabled: bool,
        boxvis_bvisd_enabled: bool,
        data_name: str,
        # inference
        video_unified_inference_enable: bool,
        prompt_as_queries: bool,
        zero_shot_inference: bool,
        test_topk_per_image: int,
        tracker_type: str,
        is_multi_cls: bool,
        apply_cls_thres: float,
        merge_on_cpu: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            boxvis_enabled: if True, use only box-level annotation; otherwise pixel-wise annotations
            boxvis_ema_enabled: if True, use Teacher Net to produce high-quality pseudo masks
            boxvis_bvisd_enabled: if True, use box-supervised VIS dataset (BVISD), including
                pseudo video clip from COCO, videos from YTVIS21 and OVIS.
        """
        super().__init__()

        # Student Net
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.text_prompt_encoder = text_prompt_encoder
        self.inference_img_generic_seg = inference_img_generic_seg 
        self.inference_video_vis = inference_video_vis            # mdqe
        self.inference_video_vis_fast = inference_video_vis_fast  # minvis
        self.inference_video_vps = inference_video_vps
        self.inference_video_vos = inference_video_vos
        self.inference_video_entity = inference_video_entity
        self.prepare_targets = prepare_targets

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold
        self.stability_score_thresh = stability_score_thresh
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.num_frames_window = max(num_frames_window, num_frames)
        self.num_frames_video = num_frames_video
        self.num_classes = num_classes
        self.is_coco = data_name.startswith("coco")

        # boxvis
        if boxvis_enabled and boxvis_ema_enabled:
            # Teacher Net
            self.backbone_t = copy.deepcopy(backbone)
            self.sem_seg_head_t = copy.deepcopy(sem_seg_head)
            self.gen_pseudo_mask = gen_pseudo_mask
            self.backbone_t.requires_grad_(False)
            self.sem_seg_head_t.requires_grad_(False)
            self.ema_shadow_decay = 0.999

        self.boxvis_enabled = boxvis_enabled
        self.boxvis_ema_enabled = boxvis_ema_enabled
        self.boxvis_bvisd_enabled = boxvis_bvisd_enabled
        self.data_name = data_name
        self.tracker_type = tracker_type  # if 'ovis' in data_name and use swin large backbone => "mdqe"

        # additional args reference
        self.video_unified_inference_enable = video_unified_inference_enable
        self.prompt_as_queries = prompt_as_queries
        self.zero_shot_inference = zero_shot_inference
        self.test_topk_per_image = test_topk_per_image
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres
        self.merge_on_cpu = merge_on_cpu   

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        lang_encoder = None
        if cfg.MODEL.UniVS.LANGUAGE_ENCODER_ENABLE and cfg.MODEL.UniVS.TEXT_PROMPT_ENCODER:
            lang_encoder = build_clip_language_encoder(cfg)
            # only eval
            for p in lang_encoder.parameters(): p.requires_grad = False
            lang_encoder.eval()
        text_prompt_encoder = None
        if cfg.MODEL.UniVS.TEXT_PROMPT_ENCODER:
            text_prompt_encoder = TextPromptEncoder(
                lang_encoder=lang_encoder,
                num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
            )
        prepare_targets = PrepareTargets(
            num_frames = cfg.INPUT.SAMPLING_FRAME_NUM,
            max_num_masks = cfg.MODEL.UniVS.NUM_POS_QUERIES,
            text_prompt_enable = cfg.MODEL.UniVS.TEXT_PROMPT_ENCODER,
            clip_class_embed_path = cfg.MODEL.UniVS.CLIP_CLASS_EMBED_PATH,
        )

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        reid_weight = cfg.MODEL.MASK_FORMER.REID_WEIGHT 
        proj_weight = dice_weight
        pair_weight = 1.
        if cfg.MODEL.BoxVIS.BoxVIS_ENABLED:
            dice_weight, mask_weight = 0.5*dice_weight, 0.5*mask_weight
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                           "loss_mask_proj": proj_weight, "loss_mask_pair": pair_weight}
        else:
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, 
                           "loss_reid": reid_weight, "loss_reid_aux": reid_weight, 
                           "loss_reid_l2p": reid_weight, "loss_reid_l2p_aux": reid_weight,
                           "loss_reid_interclip": reid_weight, "loss_reid_interclip_aux": reid_weight,
                           "loss_l2v_attn_weight": 1.0,
                           }

        class_weight_matcher = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT_MATCHER
        dice_weight_matcher = cfg.MODEL.MASK_FORMER.DICE_WEIGHT_MATCHER
        mask_weight_matcher = cfg.MODEL.MASK_FORMER.MASK_WEIGHT_MATCHER
        reid_weight_matcher = cfg.MODEL.MASK_FORMER.REID_WEIGHT_MATCHER
        # building criterion
        matcher = VideoHungarianMatcherUni(
            cost_class=class_weight_matcher,
            cost_mask=mask_weight_matcher,
            cost_dice=dice_weight_matcher,
            cost_proj=proj_weight,
            cost_pair=pair_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            boxvis_enabled=cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            boxvis_ema_enabled=cfg.MODEL.BoxVIS.EMA_ENABLED,
        )

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if cfg.MODEL.BoxVIS.EMA_ENABLED:
            gen_pseudo_mask = BoxVISTeacherSetPseudoMask(matcher)
        else:
            gen_pseudo_mask = None

        losses = ["labels", "masks", "reid"]
        criterion = VideoSetCriterionUni(
            cfg,
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )
        num_classes = sem_seg_head.num_classes

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "text_prompt_encoder": text_prompt_encoder,
            "inference_img_generic_seg": InferenceImageGenericSegmentation(cfg),
            "inference_video_vis": InferenceVideoVIS(cfg),
            "inference_video_vis_fast": InferenceVideoVISFast(cfg),
            "inference_video_vps": InferenceVideoVPS(cfg),
            "inference_video_vos": InferenceVideoVOS(cfg),
            "inference_video_entity": InferenceVideoEntity(cfg),
            "prepare_targets": prepare_targets,
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "stability_score_thresh": cfg.MODEL.MASK_FORMER.TEST.STABILITY_SCORE_THRESH,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_frames_window": cfg.INPUT.SAMPLING_FRAME_WINDOE_NUM,
            "num_frames_video": cfg.INPUT.SAMPLING_FRAME_VIDEO_NUM,
            "num_classes": num_classes,
            # boxvis
            "gen_pseudo_mask": gen_pseudo_mask,
            'boxvis_enabled': cfg.MODEL.BoxVIS.BoxVIS_ENABLED,
            "boxvis_ema_enabled": cfg.MODEL.BoxVIS.EMA_ENABLED,
            "boxvis_bvisd_enabled": cfg.MODEL.BoxVIS.BVISD_ENABLED,
            "data_name": cfg.DATASETS.TEST[0],
            # inference
            "video_unified_inference_enable": cfg.MODEL.UniVS.TEST.VIDEO_UNIFIED_INFERENCE_ENABLE,
            "prompt_as_queries": cfg.MODEL.UniVS.PROMPT_AS_QUERIES,
            "zero_shot_inference": cfg.MODEL.BoxVIS.TEST.ZERO_SHOT_INFERENCE,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "tracker_type": cfg.MODEL.BoxVIS.TEST.TRACKER_TYPE,
            "is_multi_cls": cfg.MODEL.BoxVIS.TEST.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.BoxVIS.TEST.APPLY_CLS_THRES,
            "merge_on_cpu": cfg.MODEL.BoxVIS.TEST.MERGE_ON_CPU,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]: each dict has the results for one image.
        """
        if not self.training:
            dataset_name = batched_inputs[0]["dataset_name"]
            if dataset_name.startswith("coco") or dataset_name.startswith("ade20k"):
                # evaluation for images
                raise ValueError(f"Not support to eval the image datasets {dataset_name} here")
            else:
                # evaluation for videos
                if self.video_unified_inference_enable:
                    if dataset_name.startswith("ytvis") or dataset_name.startswith("ovis") \
                         or dataset_name.startswith("vipseg") or dataset_name.startswith("vpsw"):
                        return self.inference_video_entity.eval(self, batched_inputs)
                    else:
                        raise ValueError(f"Not support to eval the dataset {dataset_name} yet")
                else:
                    if dataset_name.startswith("ytvis") or dataset_name.startswith("ovis"):
                        if self.tracker_type == 'mdqe':  # mdqe
                            return self.inference_video_vis.eval(self, batched_inputs)
                        else:  # minvis
                            return self.inference_video_vis_fast.eval(self, batched_inputs)
                    elif dataset_name.startswith("vipseg") or dataset_name.startswith("vpsw"):
                        return self.inference_video_vps.eval(self, batched_inputs)
                    elif dataset_name.startswith("sot"):
                        return self.inference_video_vos.eval(self, batched_inputs)
                    else:
                        raise ValueError(f"Not support to eval the dataset {dataset_name} yet")

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        images = ImageList.from_tensors(images, self.size_divisibility)
        images_tensor = images.tensor

        video_len = images_tensor.shape[0]
        assert video_len == self.num_frames_video, f"dismatch the video length: {video_len} and {self.num_frames_video}, please check!"

        targets_entire_video = self.prepare_targets.process(batched_inputs, images, self.device, self.text_prompt_encoder)
        assert len(targets_entire_video) == 1, "only support a video once time!"

        is_last = False
        stride = max(self.num_frames - 1, 1)
        for i in range(0, video_len, stride):
            if i + self.num_frames >= video_len:
                is_last = True

            e_idx = min(video_len, i + self.num_frames)  
            i = e_idx - self.num_frames       
            features = self.backbone(images_tensor[i:e_idx])
            pixel_decoder_out_tuple = self.sem_seg_head.pixel_decoder.forward_features(features)
            
            mask_features = pixel_decoder_out_tuple[0]
            mask_features_bfe_conv = pixel_decoder_out_tuple[1]
            multi_scale_features = pixel_decoder_out_tuple[-1]

            targets = self.slice_targets_per_clip(i, targets_entire_video)
            
            outputs = self.sem_seg_head.predictor(
                multi_scale_features, mask_features, mask_features_bfe_conv, targets=targets
            )

            losses_i = self.criterion(outputs, targets)
            if i == 0:
                losses = {k: [] for k in losses_i.keys()}
            for k, v in losses_i.items():   
                losses[k].append(v)
            
            losses_interclip = self.interclip_reid_loss(is_last, targets, targets_entire_video)
            
            with torch.no_grad():
                if not is_last and (targets[0]['prompt_type'] == 'visual' or targets[0]['task'] == 'grounding'):
                    self.prepare_prompt_memory_pool(i, multi_scale_features, targets, targets_entire_video)

        losses.update(losses_interclip)
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                if isinstance(losses[k], list):
                    losses[k] = sum(losses[k])/max(len(losses[k]), 1) 
                losses[k] = losses[k] * self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                print(f"Loss {k} not in weight_dict, remove it.")
                losses.pop(k)

        return losses
    
    def interclip_reid_loss(self, is_last, targets, targets_entire_video):
        targets_per_video = targets_entire_video[0] 
        if 'src_embds' not in targets_per_video:
            targets_per_video['src_embds'] = [[] for _ in range(10)] 
            targets_per_video['tgt_ids'] = [[] for _ in range(10)] 

        targets_per_clip =  targets[0]
        for l in range(len(targets_per_clip['src_embds'])):
            dim_len = len(targets_per_clip['src_embds'][l])
            assert dim_len <= 2, f"The dimension should be 1 or 2, but get{dim_len}"
            targets_per_video['src_embds'][l] += targets_per_clip['src_embds'][l]
            targets_per_video['tgt_ids'][l] += targets_per_clip['tgt_ids'][l]
        
        if is_last:
            losses = {}
            num_layers = len(targets_per_video['src_embds'])
            for l in range(num_layers):
                if l == 0:
                    continue
                
                src_embds = torch.cat(targets_per_video['src_embds'][l])
                tgt_ids = torch.cat(targets_per_video['tgt_ids'][l])

                rdm_idxs = torch.randperm(len(tgt_ids))
                src_embds = src_embds[rdm_idxs]
                tgt_ids = tgt_ids[rdm_idxs]

                tgt_ids_unique = tgt_ids.unique()
                tgt_ids_unique = tgt_ids_unique[tgt_ids_unique >= 0]
                tgt_ids_unique = tgt_ids_unique[~torch.isinf(tgt_ids_unique)]

                if tgt_ids_unique.nelement() == 0:
                    loss_ctt = src_embds[:0].sum().detach()
                    loss_aux = src_embds[:0].sum().detach()
                else:
                    src_idxs_unique = (tgt_ids_unique[:, None] == tgt_ids[None]).float().argmax(-1).reshape(-1)
                    same_tgt_ids = (tgt_ids_unique[:, None] == tgt_ids[None]).float()

                    # ctt loss
                    sim = torch.einsum('pc,kc->pk', src_embds[src_idxs_unique], src_embds) / math.sqrt(src_embds.shape[-1])
                    loss_ctt = contrastive_loss(sim, same_tgt_ids)

                    # aux loss
                    src_embds_norm = F.normalize(src_embds, p=2, dim=-1)
                    sim_aux = torch.einsum('pc,kc->pk', src_embds_norm[src_idxs_unique], src_embds_norm)
                    loss_aux = contrastive_aux_loss(sim_aux, same_tgt_ids)

                if l == num_layers-1:
                    losses['loss_reid_interclip'] = loss_ctt
                    losses['loss_reid_interclip_aux'] = loss_aux
                else:
                    losses['loss_reid_interclip_'+str(l)] = loss_ctt
                    losses['loss_reid_interclip_aux_'+str(l)] = loss_aux
            
            return losses
        
        else:
            return None

    def slice_targets_per_clip(self, first_frame_idx, targets_entire_video):
        targets = []
        for targets_per_video in targets_entire_video:
            targets_per_clip = {}
            for k, v in targets_per_video.items():
                if k in {"ids", "masks", "boxes", "sem_masks"}:
                    targets_per_clip[k] = v[:, first_frame_idx:first_frame_idx+self.num_frames]
                elif k in {"frame_indices"}:
                    targets_per_clip[k] = v[first_frame_idx:first_frame_idx+self.num_frames]
                elif k not in {"src_embds", "tgt_ids"}:
                    targets_per_clip[k] = v
            targets.append(targets_per_clip)

        return targets
    
    def prepare_prompt_memory_pool(self, first_frame_idx, x, targets, targets_entire_video):
        """
        x: multi_scale_features
        prompt_embds: Q_l+Q_p, L, T, C
        """
        if "prompt_feats" in targets_entire_video[0]:
            return

        bs = len(targets)
        assert len(targets) == 1, 'Only support batch size = 1'
        t = x[0].shape[0] // bs

        # x is a list of multi-scale feature
        src = []
        pos = []
        size_list = []
        for i in range(len(x)):
            size_list.append(x[i].shape[-2:])
            pos.append(self.sem_seg_head.predictor.pe_layer(x[i].view(bs, t, -1, size_list[-1][0], size_list[-1][1]), None).flatten(3))
            src.append(self.sem_seg_head.predictor.input_proj[i](x[i]).flatten(2) + self.sem_seg_head.predictor.level_embed.weight[i][None, :, None])

            # NTxCxHW => HWxNTxC
            pos[-1] = pos[-1].flatten(0, 1).permute(2,0,1)
            src[-1] = src[-1].permute(2,0,1)
        
        prompt_tuple = self.sem_seg_head.predictor.forward_prompt_encoder(
            src, pos, size_list, targets, num_frames=t, prompt_type='visual', use_all_prev_frames=first_frame_idx==0
        )
        prompt_feats, prompt_pe, prompt_self_attn_masks = prompt_tuple[2:5]
        if prompt_feats is None:
            return 

        targets_per_video = targets_entire_video[0]
        if "prompt_feats" not in targets_per_video:
            targets_per_video["prompt_feats"] = prompt_feats
            targets_per_video["prompt_pe"] = prompt_pe
            targets_per_video["prompt_self_attn_masks"] = prompt_self_attn_masks
    
        else:
            targets_per_video["prompt_feats"] = torch.cat(
                [targets_per_video["prompt_feats"], prompt_feats], dim=1
            )
            targets_per_video["prompt_pe"] = torch.cat(
                [targets_per_video["prompt_pe"], prompt_pe], dim=1
            )
            targets_per_video["prompt_self_attn_masks"] = None 
            
    


