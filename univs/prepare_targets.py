import os
import torch
import torch.nn as nn

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.data import MetadataCatalog

from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info
from univs.modeling.language import pre_tokenize_expression
from univs.utils.comm import box_xyxy_to_cxcywh, convert_box_to_mask


def is_semseg_dataset(dataset_name):
    if dataset_name.startswith("vspw"):
        return True
    else:
        return False


class PrepareTargets:
    def __init__(
        self,
        num_frames: int=1,
        max_num_masks: int=30,
        text_prompt_enable: bool=False,
        boxvis_enabled: bool=False,
        clip_class_embed_path: str="",
        thing_only_enabled: bool=False,
        semantic_on: bool=False,
    ):
        self.num_frames = num_frames
        self.max_num_masks = max_num_masks
        self.text_prompt_enable = text_prompt_enable
        self.boxvis_enabled = boxvis_enabled

        # convert video category to text embeddings (offline generation)
        self.clip_cls_text_emb = torch.load(clip_class_embed_path)
        self.thing_only_enabled = thing_only_enabled
        self.semantic_on = semantic_on
    
    def process_inference(self, targets, inter_image_size, device, text_prompt_encoder=None, image_size=None):
        """
        convert coco-format annotations to required format during inference
        inter_image_size: the size of padding images 
        """
        task = targets[0]["task"]
        if task == "grounding":
            prompt_type = 'text'
        elif task == "detection":
            prompt_type = 'text' if is_semseg_dataset(targets[0]["dataset_name"]) or self.semantic_on else 'visual'
        else:
            prompt_type = 'visual'  # sot task

        clip_gt_instances = []
        for targets_per_video in targets:
            clip_gt_instances.append(
                {
                    "video_len": targets_per_video["video_len"], 
                    "dataset_name": targets_per_video["dataset_name"],
                    "task": targets_per_video["task"],
                    "num_frames": self.num_frames,
                    "inter_image_size": inter_image_size,
                    "image_size": image_size,
                    "file_names":  targets_per_video["file_names"],
                }
            )

            targets_per_video["prompt_type"] = prompt_type
            clip_gt_instances[-1]["prompt_type"] = prompt_type
            if "mask_palette" in targets_per_video:
                clip_gt_instances[-1]["mask_palette"] = targets_per_video["mask_palette"]

            if task == "sot":
                clip_gt_instances[-1]["instances"] = targets_per_video["instances"]

        if task in {"detection", "grounding"}:
            self.preprocess_text_prompt(
                text_prompt_encoder, targets, clip_gt_instances, device=device, is_train=False
            )

        return clip_gt_instances

    def process(self, targets, images, device, text_prompt_encoder=None, is_train=True):
        """
       convert coco-format annotations to required format during training
        """
        # Note: images without MEAN and STD normalization
        image_size = images.image_sizes[0] 
        BT, c, h_pad, w_pad = images.tensor.shape
        box_normalizer = torch.as_tensor([w_pad, h_pad, w_pad, h_pad],
                                         dtype=torch.float32, device=device).reshape(1, -1)

        if targets[0]["task"] == "grounding":
            prompt_type = 'text'
        elif targets[0]["task"] == "detection":
            if targets[0]['dataset_name'] in {'lvis', 'burst', 'flickr', "entityseg_panoptic"}:
                prompt_type = 'visual'
            else:
                prompt_type = 'visual' if torch.randn(1) > 0.5 else 'text'
        else:
            prompt_type = 'visual'  # sot task

        num_max_instances = -1
        clip_gt_instances = []
        valid_bool_clips = []
        for targets_per_video in targets:
            has_mask = targets_per_video["has_mask"]
            # object classes that occured in this video
            occur_classes = torch.stack([
                targets_per_frame.gt_classes for targets_per_frame in targets_per_video["instances"]
            ]).max(0)[0]
            occur_sem_classes = occur_classes.unique()[occur_classes.unique() >= 0]

            _num_frames = len(targets_per_video['file_names'])
            _num_instance = len(targets_per_video["instances"][0])
            # if targets_per_video['dataset_name'] in {"entityseg_panoptic"}:
            #     print('original num instances:', _num_instance)
            if _num_instance > 2*self.max_num_masks and targets_per_video['dataset_name'] in {"sa_1b", "burst", "lvis", "entityseg_panoptic"}:
                # there are so many objects in SA1B/lvis/burst datasets, we select a fixed number of objects
                # to keep memory balance on multiply GPUs
                if targets_per_video['dataset_name'] in {"sa_1b"} or occur_sem_classes.nelement() == 0:
                    slt_idxs = torch.randperm(_num_instance, device=device)[:2*self.max_num_masks]
                else:
                    slt_idxs_cls = occur_sem_classes[torch.randperm(len(occur_sem_classes))]
                    slt_idxs = []
                    for i, i_cls in enumerate(slt_idxs_cls):
                        if i > 0 and len(torch.cat(slt_idxs)) > 2*self.max_num_masks:
                            break
                        slt_idxs.append(torch.nonzero(occur_classes == i_cls).reshape(-1))
                    slt_idxs = torch.cat(slt_idxs).to(device)[:2*self.max_num_masks]
                    
                slt_idxs = slt_idxs.long()
                # to speed up training and save memory, there are so many objects in SA1B
                targets_per_video["instances"] = [
                    targets_per_frame[slt_idxs] for targets_per_frame in targets_per_video["instances"]
                ]
                _num_instance = len(slt_idxs)

            if has_mask:
                mask_shape = [_num_instance, _num_frames, h_pad, w_pad]
                gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=device)
            gt_boxes_per_video = torch.zeros([_num_instance, _num_frames, 4], dtype=torch.float32, device=device)
            gt_classes_per_video = targets_per_video["instances"][0].gt_classes.to(device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                _update_box = box_xyxy_to_cxcywh(targets_per_frame.gt_boxes.tensor)[..., 2:].gt(0).all(-1)
                gt_boxes_per_video[_update_box, f_i] = targets_per_frame.gt_boxes.tensor[_update_box] / box_normalizer  # xyxy
                
                if has_mask:
                    if isinstance(targets_per_frame.gt_masks, BitMasks):
                        gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                    else:  # polygon
                        gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            if not has_mask:
                gt_masks_per_video = convert_box_to_mask(gt_boxes_per_video, h_pad, w_pad)
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2, 3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(-1)
            if self.thing_only_enabled and targets[0]['dataset_name'] in {"vipseg", "coco_panoptic", "ade20k"}:
                if targets[0]['dataset_name'] == "ade20k":
                    raise ValueError(f"Training on thing objects only, Unsupported semantic segmetnation dataset: {targets[0]['dataset_name']}")
                else:
                    if targets[0]['dataset_name'] == "vipseg":
                        metadata = MetadataCatalog.get("vipseg_panoptic_train")
                    elif targets[0]['dataset_name'] == "coco_panoptic":
                        metadata = MetadataCatalog.get("coco_panoptic_train")
                    else:
                        raise ValueError(f" Unsupported dataset name: {targets[0]['dataset_name']}")
                    valid_is_thing = torch.tensor([
                        int(c) in metadata.thing_dataset_id_to_contiguous_id for c in gt_classes_per_video
                    ], device=valid_bool_clip.device)
                    valid_bool_clip = valid_bool_clip & valid_is_thing

            # gt_classes_per_video = -1, if dataset has not class annotations, such as SA1B
            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long()  # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()          # N, num_frames
            gt_boxes_per_video = gt_boxes_per_video[valid_bool_clip].float()     # N, num_frames, 4
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()     # N, num_frames, H, W
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            frame_indices = torch.as_tensor(targets_per_video["frame_indices"], device=device)  # num_frames
            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id  # obj id mapping
            
            assert not (gt_classes_per_video == 0).any(), 'Class labels should start from 1 instead of 0!!'
            # if targets_per_video['dataset_name'] in {"entityseg_panoptic"}:
            #     print(gt_masks_per_video.shape)
                
            valid_bool_clips.append(valid_bool_clip)
            clip_gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video,
                    "masks": gt_masks_per_video, "boxes": gt_boxes_per_video,
                    "video_len": targets_per_video["video_len"], 
                    "frame_indices": frame_indices,
                    "occur_sem_labels": occur_sem_classes,
                    "dataset_name": targets_per_video["dataset_name"],
                    "task": targets_per_video["task"],
                    "num_frames": _num_frames,
                    "has_mask": has_mask,
                    "has_stuff": targets_per_video["has_stuff"]
                }
            )
            
            del gt_masks_per_video
            # if targets_per_video["dataset_name"] in {'flickr'}
                # clip_gt_instances[-1]["token_ids"] = [targets_per_frame.gt_token_ids for targets_per_frame in targets_per_video]

            targets_per_video["prompt_type"] = prompt_type
            clip_gt_instances[-1]["prompt_type"] = prompt_type
            if targets_per_video["task"] == "detection" and prompt_type == 'text':
                # number of semantic categories, instead of number of instances
                if len(gt_classes_per_video.unique()) > num_max_instances:
                    num_max_instances = len(gt_classes_per_video.unique())
            else:
                if _num_instance > num_max_instances:
                    num_max_instances = _num_instance
        
        if self.text_prompt_enable:
            assert text_prompt_encoder is not None
            self.preprocess_text_prompt(
                text_prompt_encoder, targets, clip_gt_instances, valid_bool_clips, device, num_max_instances
            )
        
        # use part of GT instances to avoid out of memory errors
        # eg. for detection task, vipseg has masks with shape [69, 2, 608, 1024] => Out of memory
        for k, (gt_instances, targets_per_video) in enumerate(zip(clip_gt_instances, targets)):
            if gt_instances["masks"].shape[0] > self.max_num_masks:
                gt_instances["labels"] = gt_instances["labels"][:self.max_num_masks]
                gt_instances["ids"] = gt_instances["ids"][:self.max_num_masks]
                gt_instances["masks"] = gt_instances["masks"][:self.max_num_masks]
                gt_instances["boxes"] = gt_instances["boxes"][:self.max_num_masks]

        return clip_gt_instances
    
    def preprocess_text_prompt(
        self, text_prompt_encoder, targets, clip_gt_instances, valid_bool_clips=None, 
        device='cpu', num_max_instances=30, is_train=True
    ):
        for k, (gt_instances, targets_per_video) in enumerate(zip(clip_gt_instances, targets)):
            if is_train:
                num_max_instances = min(num_max_instances, self.max_num_masks)
                gt_instances["num_max_instances"] = num_max_instances
            
            if targets_per_video["task"] == "grounding":
                has_expression = ("expressions" in targets_per_video) and (len(targets_per_video["expressions"]) > 0)
                if has_expression and (not is_train or valid_bool_clips[k].sum()):
                    expressions = targets_per_video["expressions"]  # list[exp1, exp2, ...]
                    exp_obj_ids = targets_per_video["exp_obj_ids"]  # list[115, 115, 116, 116, ...], an object has multiple expressions in RefYTVOS datasets 

                    assert len(expressions) == len(exp_obj_ids), \
                        f'Mismatch number between expressions and exp_ids: {len(expressions)} and {len(exp_obj_ids)}'
                    
                    if is_train:
                        expressions = [expressions[e_i] for e_i, valid in enumerate(valid_bool_clips[k]) if valid]
                        exp_obj_ids = [exp_obj_ids[e_i] for e_i, valid in enumerate(valid_bool_clips[k]) if valid]
                        # original exp_obj_ids -> zero-based ids
                        e_i, exp_obj_ids_map = 0, {}
                        for _obj_id in exp_obj_ids:
                            if _obj_id not in exp_obj_ids_map:
                                exp_obj_ids_map[_obj_id] = e_i
                                e_i = e_i + 1
                        exp_obj_ids = torch.as_tensor(
                            [exp_obj_ids_map[_obj_id] for _obj_id in exp_obj_ids], dtype=torch.int64, device=device
                        )
                        # keep consistency with ids and exp_obj_ids, exp_obj_id it id != 1 else -1
                        gt_instances["ids"] = exp_obj_ids.view(-1,1) * (gt_instances["ids"] != -1) - (gt_instances["ids"] == -1).float()

                    gt_instances["expressions"] = expressions
                    gt_instances["exp_obj_ids"] = exp_obj_ids
                    exp_word_feats, exp_sentence_feats, len_word_expressions = \
                        text_prompt_encoder.get_expression_prompt(expressions, device)
                    
                    if is_train and exp_sentence_feats.shape[0] < num_max_instances:
                        exp_word_feats = exp_word_feats.repeat(num_max_instances,1,1,1)
                        exp_sentence_feats = exp_sentence_feats.repeat(num_max_instances,1,1)
                        exp_obj_ids = exp_obj_ids.repeat(num_max_instances)
                        len_word_expressions = (len_word_expressions * num_max_instances)
                        
                else:
                    if not is_train:
                        continue

                    exp_word_feats = torch.zeros((num_max_instances, 77, self.num_frames, 640), device=device)
                    exp_sentence_feats = torch.zeros((num_max_instances, self.num_frames, 640), device=device)
                    exp_obj_ids = torch.ones(num_max_instances, dtype=torch.int64, device=device) * -1
                    len_word_expressions = [0] * num_max_instances
                    gt_instances["exp_obj_ids"] = exp_obj_ids

                len_word_expressions = len_word_expressions[:num_max_instances]
                exp_word_feats = exp_word_feats[:num_max_instances]
                exp_sentence_feats = exp_sentence_feats[:num_max_instances]
                exp_obj_ids = exp_obj_ids[:num_max_instances]
                
                gt_instances["exp_word_len"] = len_word_expressions
                gt_instances["exp_word_feats"] = exp_word_feats
                gt_instances["exp_sentence_feats"] = exp_sentence_feats
                gt_instances["prompt_obj_ids"] = exp_obj_ids

            elif targets_per_video["task"] == "detection":
                if targets_per_video["dataset_name"] not in {'flickr'}:
                    dataset_name = targets_per_video["dataset_name"]
                    if targets_per_video["is_raw_video"] or (not is_train and dataset_name not in combined_datasets_category_info):
                        clip_cls_text_emb = self.clip_cls_text_emb
                    else:
                        if dataset_name not in combined_datasets_category_info:
                            raise ValueError(f"Unsupported dataset_name: {dataset_name}")
                        num_classes, start_idx = combined_datasets_category_info[dataset_name]
                        clip_cls_text_emb = self.clip_cls_text_emb[start_idx:start_idx+num_classes]
                else:
                    clip_cls_text_emb = text_prompt_encoder.get_expression_prompt(
                        targets_per_video["phrases"], device, text_type='class_name'
                    )[1]
                    clip_cls_text_emb = clip_cls_text_emb[:, 0]  # n, t ,c -> n, c
                    token_ids = targets_per_video['token_ids']
                    if is_train:
                        token_ids = [token_ids[e_i] for e_i, valid in enumerate(valid_bool_clips[k]) if valid]
                    gt_instances["token_ids"] = token_ids

                if is_train and targets_per_video["prompt_type"] == 'text':
                    gt_sem_labels_per_video = gt_instances["labels"].unique()
                    is_same_labels = (gt_sem_labels_per_video[:, None] == gt_instances["labels"][None]).float()
                    gt_sem_masks_per_video = torch.einsum('kn,nthw->kthw', is_same_labels, gt_instances["masks"]).gt(0.)
                    gt_instances["sem_labels"] = gt_sem_labels_per_video
                    gt_instances["sem_masks"] = gt_sem_masks_per_video

                    # labels that not appeared in this video
                    if self.thing_only_enabled and targets_per_video['dataset_name'] in {"vipseg", "coco_panoptic"}:
                        if targets_per_video['dataset_name'] == "vipseg":
                            metadata = MetadataCatalog.get("vipseg_panoptic_train")
                        elif targets_per_video['dataset_name'] == "coco_panoptic":
                            metadata = MetadataCatalog.get("coco_panoptic_train")
                        else:
                            raise ValueError(f" Unsupported dataset name: {targets_per_video['dataset_name']}")
                        no_object_gt_labels = torch.as_tensor([
                            l for l in range(1, num_classes+1) 
                            if l not in gt_instances["occur_sem_labels"] and l in metadata.thing_dataset_id_to_contiguous_id
                        ])  # Note: start from 1 instead of 0
                    else:
                        no_object_gt_labels = torch.as_tensor([
                            l for l in range(1, num_classes+1) if l not in gt_instances["occur_sem_labels"]
                        ])  # Note: start from 1 instead of 0
                    # select a subset of object categories, as there are too many categories in lvis/burst/objects365
                    # "no_object_labels": labels that not appear in this video
                    # "sem_labels": thing and stuff labels that appear in this video
                    no_object_gt_labels = no_object_gt_labels[torch.randperm(len(no_object_gt_labels))].to(device)
                    num_padding = max(num_max_instances - len(gt_sem_labels_per_video), 0)
                    if num_padding > 0:
                        no_object_gt_labels = no_object_gt_labels.repeat(num_padding)
                    no_object_gt_labels = no_object_gt_labels[:num_padding]
                    gt_instances["prompt_obj_ids"] = torch.cat([
                        torch.arange(len(gt_sem_labels_per_video), device=device), 
                        torch.ones_like(no_object_gt_labels) * -1
                    ])[:num_max_instances]  # semantic ids 
                    gt_instances["prompt_gt_labels"] = torch.cat([
                        gt_sem_labels_per_video, no_object_gt_labels
                    ])[:num_max_instances]  # semantic ids
                    gt_instances["clip_cls_text_emb"] = clip_cls_text_emb[gt_instances["prompt_gt_labels"]-1]
                
                else:
                    gt_instances["clip_cls_text_emb"] = clip_cls_text_emb
    
    