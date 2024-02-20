import torch
import torch.nn.functional as F
import scipy.cluster.hierarchy as hac

from einops import rearrange, repeat
from detectron2.projects.point_rend.point_features import point_sample

from univs.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingSine3DArbitraryT
from univs.modeling.language import pre_tokenize_expression
from univs.utils.comm import (
    convert_box_to_mask,
    convert_mask_to_box,
    box_xyxy_to_cxcywh,
)

class TextPromptEncoder:
    def __init__(
            self,
            lang_encoder,
            num_frames,
    ):
        self.lang_encoder = lang_encoder
        if lang_encoder is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lang_encoder = lang_encoder.to(device)
        self.num_frames = num_frames

    def get_expression_prompt(self, expressions, device):
        """
        Args:
            expressions: List[exp_obj1, exp_obj2, ...]
            device:

        Returns:

        """
        assert self.lang_encoder is not None, 'No language encoder is assigned!!'
        # input expression into CLIP text encoder to obtain exp embedding
        # num_exp x 81 x 77, where 81 and 77 are the number of templates and the length of exp
        # TODO: language model uses byte to split sentences, not word by word
        len_word_expressions = [len(exp.split(' ')) + 5 for exp in expressions]
        exp_tokenizers = pre_tokenize_expression(expressions).to(device)
        exp_word_feats, exp_sentence_feats = [], []
        for exp_tokenizer in exp_tokenizers:
            exp_feats = self.lang_encoder.encode_text(exp_tokenizer, only_eot=False)
            exp_word_feats.append(exp_feats[0][0])  # 81x77x640 -> 77x640
            exp_sentence_feats.append(exp_feats[1].mean(0))  # 81x640 -> 640
        # exp_word_feats: num_exp x 77 x 640, exp_sentence_feats: num_exp x 640
        exp_word_feats = torch.stack(exp_word_feats)[:, :, None].repeat(1, 1, self.num_frames, 1)
        exp_sentence_feats = torch.stack(exp_sentence_feats)[:, None].repeat(1, self.num_frames, 1)

        return exp_word_feats, exp_sentence_feats, len_word_expressions


class VisualPromptEncoder:
    def __init__(
            self,
            pretrain_img_size=1024,
            hidden_dim=256,
            num_frames=1,
            num_dense_points=32,
            position_embedding_sin3d_type="FixedT",
    ):
        # positional encoding
        N_steps = hidden_dim // 2
        if position_embedding_sin3d_type == "FixedT":
            self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)
        else:
            self.pe_layer = PositionEmbeddingSine3DArbitraryT(N_steps, normalize=True)
        self.num_frames = num_frames
        self.pretrain_img_size = pretrain_img_size
        self.num_dense_points = num_dense_points
        self.position_embedding_sin3d_type = position_embedding_sin3d_type

        self.key_fid = int((num_frames - 1) / 2)
        self.img_feats_scale = 8  # the last one of multi-scale feaures has 1/8 resolution of input image

    @torch.no_grad()
    def get_point_prompt(
        self, img_features, img_pos, point_coords=None, boxes=None, masks=None, 
        key_fid=None, key_fid_original=None, is_train=False, enable_dense_prompt=True,
    ):
        """
        Args:
            point_coords: Q x 2
            boxes: None or Q x 4
            masks: None or Q x H x W
            img_features: C x H/s x W/s, s is the scale
            img_pos: C x H/s x W/s

        Returns:
            point_coords: Q x 2
            query_pe: Q x L x T x C
            query_feats: Q x L x T x C, directly repeated features in temporal direction
            query_attn_masks: T x 1 x Q x (HW/16**2)
        """
        key_fid = self.key_fid if key_fid is None else key_fid
        if key_fid_original is None:
            key_fid_original = key_fid

        h_img, w_img = img_features.shape[-2:]

        if point_coords is not None:
            assert point_coords.dim() == 2
        else:
            assert boxes is not None or masks is not None
            point_coords = self.select_points_from_box_mask(h_img, w_img, masks=masks, boxes=boxes)
        
        valid = ((point_coords >= 0) & ((1-point_coords) >= 0)).sum(-1) == 2
        point_coords = point_coords * valid.float().view(-1,1) 
            
        # generate positional embeddings for points
        device = point_coords.device
        _num_insts, C = point_coords.shape

        input_size = (self.num_frames, h_img*self.img_feats_scale, w_img*self.img_feats_scale)
        if self.position_embedding_sin3d_type == "FixedT":
            # T, Q, C -> Q, T, C
            query_pe = self.pe_layer.forward_points_with_size(input_size, point_coords)
            query_pe = query_pe[key_fid].unsqueeze(1).repeat(1,self.num_frames,1)
        else:
            query_pe = self.pe_layer.forward_points_with_size(input_size, point_coords, key_fid_original)
            query_pe = query_pe.transpose(0, 1)

        point_feats = point_sample(
            img_features.unsqueeze(0),
            point_coords.unsqueeze(0),
            align_corners=False
        )  # 1, C, Q
        query_feats = point_feats.permute(2, 0, 1).repeat(1, self.num_frames, 1)

        blank_attn_masks = torch.zeros(
            (self.num_frames, 1, 1, h_img*w_img)
        ).bool()
        # four pixels for per point-prompt with the coords (x, y)
        query_attn_masks = (~blank_attn_masks).repeat(1, 1, _num_insts, 1).to(device)
        point_coords_wh = point_coords * torch.as_tensor([w_img, h_img], device=device).view(1,-1) 
        point_coords_wh_floor = point_coords_wh.floor().long()
        indices = point_coords_wh_floor[:, 1] * w_img + point_coords_wh_floor[:, 0]
        query_attn_masks[key_fid, :, range(len(indices)), indices] = False
        point_coords_wh_ceil = point_coords_wh.ceil().long()
        indices = point_coords_wh_ceil[:, 1] * w_img + point_coords_wh_ceil[:, 0]
        indices = indices.clamp(min=0, max=w_img*h_img-1)
        query_attn_masks[key_fid, :, range(len(indices)), indices] = False
        indices = point_coords_wh_floor[:, 1] * w_img + point_coords_wh_ceil[:, 0]
        indices = indices.clamp(min=0, max=w_img*h_img-1)
        query_attn_masks[key_fid, :, range(len(indices)), indices] = False
        indices = point_coords_wh_ceil[:, 1] * w_img + point_coords_wh_floor[:, 0]
        indices = indices.clamp(min=0, max=w_img*h_img-1)
        query_attn_masks[key_fid, :, range(len(indices)), indices] = False

        query_feats_dense = query_feats[:, None]
        query_pe_dense = query_pe[:, None]
        if enable_dense_prompt:
            query_feats_dense = query_feats_dense.repeat(1, self.num_dense_points, 1, 1)
            query_pe_dense =query_pe_dense.repeat(1, self.num_dense_points, 1, 1)
        
        if (~valid).any():  # non-appeared objects in the key frame
            query_pe_dense = query_pe_dense * valid.view(-1,1,1,1)
            query_feats_dense = query_feats_dense * valid.view(-1,1,1,1)
            query_attn_masks[:, :, ~valid] = False
        
        return point_coords, query_pe_dense, query_feats_dense, query_attn_masks

    @torch.no_grad()
    def get_mask_prompt(
        self, img_features, img_pos, masks, boxes=None, mask_thresh=0.5, 
        key_fid=None, key_fid_original=None, is_train=False, enable_dense_prompt=True
    ):
        """

        Args:
            img_features: C x H/16 x W/16
            img_pos: C x H/16 x W/16
            masks: Q x H_img x W_img, binary masks or masks after sigmoid function, value in [0, 1]
            boxes: Q x 4
            is_train: during training or inference
            mask_thresh: a threshold to extract the instance masks
            enable_dense_prompt:

        Returns:
            point_coords: Q x 2
            query_pe: Q x L x T x C
            query_feats: Q x L x T x C, directly repeated features in temporal direction
            query_attn_masks: T x 1 x Q x (HW/16**2)

        """
        key_fid = self.key_fid if key_fid is None else key_fid
        if key_fid_original is None:
            key_fid_original = key_fid

        h_img, w_img = img_features.shape[-2:]
        device = img_features.device

        assert masks.dim() == 3, f'Mask shape shoule be num_instsxHxW, but get {masks.shape}'
        valid = masks.gt(mask_thresh).flatten(1).sum(-1) > 0

        _num_insts, h, w = masks.shape
        point_coords = self.select_points_from_box_mask(h_img, w_img, masks=masks, boxes=boxes)

        # generate positional embeddings for points: Q, C -> Q, T, C
        input_size = (self.num_frames, h_img*self.img_feats_scale, w_img*self.img_feats_scale)
        if self.position_embedding_sin3d_type == "FixedT":
            query_pe = self.pe_layer.forward_points_with_size(input_size, point_coords)
            query_pe = query_pe[key_fid].unsqueeze(1).repeat(1,self.num_frames,1)
        else:
            query_pe = self.pe_layer.forward_points_with_size(input_size, point_coords, key_fid_original)
            query_pe = query_pe.transpose(0, 1)

        img_masks = torch.zeros(
            (_num_insts, h_img*self.img_feats_scale, w_img*self.img_feats_scale), device=masks.device
        )
        img_masks[:, :h, :w] = masks.float()  # Q, H, W
        feat_masks = F.interpolate(
            img_masks.unsqueeze(1),
            (img_features.shape[-2], img_features.shape[-1]),
            mode='nearest',
        ).squeeze(1)
        feat_masks_binary = feat_masks >= min(mask_thresh, feat_masks.max())

        # weighted features on all points in the mask as point_feats
        feat_masks_w = feat_masks * feat_masks_binary
        point_feats_key = torch.einsum('qn,nc->qc', 
                                       feat_masks_w.flatten(-2).float(),
                                       img_features.flatten(-2).t())
        point_feats_key = point_feats_key / feat_masks_w.sum((-2, -1)).clamp(min=mask_thresh)[:, None]
        query_feats = point_feats_key[:, None].repeat(1, self.num_frames, 1)

        if boxes is None:
            normlizer = torch.tensor([
                w_img*self.img_feats_scale, h_img*self.img_feats_scale, w_img*self.img_feats_scale, h_img*self.img_feats_scale
            ]).reshape(1, -1)
            boxes_wo_normalized = convert_mask_to_box(masks > mask_thresh)
            boxes = boxes_wo_normalized / normlizer

        if is_train and feat_masks_binary.sum() > 16:
            boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
            boxes_cxcy, boxes_wh = boxes_cxcywh[:, :2], boxes_cxcywh[:, 2:]
            offsets = torch.rand(boxes_wh.shape).to(device) * 2 - 1  # [-1, 1]
            # coords in [cx-0.25*w, cx+0.25w] x [cy-0.25*h, cy+0.25*h]
            boxes_wh = (1 + offsets * 0.25) * boxes_wh  # Q, 2
            boxes = torch.cat([boxes_cxcy - 0.5 * boxes_wh, boxes_cxcy + 0.5 * boxes_wh], dim=-1)
            boxes = boxes.clamp(min=0, max=1)
        
        query_attn_masks = torch.zeros((self.num_frames, 1, _num_insts, h_img*w_img)).bool().to(device)
        box_attn_masks = convert_box_to_mask(boxes, h_img, w_img)
        query_attn_masks[key_fid, 0] = torch.logical_not(box_attn_masks.flatten(-2))

        query_feats_dense = query_feats[:, None]
        query_pe_dense = query_pe[:, None]
        if enable_dense_prompt:
            query_feats_dense, query_pe_dense = self.get_dense_features(
                img_features, img_pos, feat_masks_binary, query_pe, query_feats, prompt_type="masks", is_train=is_train
            )
        
        if (~valid).any():  # non-appeared objects in the key frame
            query_p_densee = query_pe_dense * valid.view(-1,1,1,1).float()
            query_feats_dense = query_feats_dense * valid.view(-1,1,1,1).float()
            query_attn_masks[:, :, ~valid] = False

        return point_coords, query_pe_dense, query_feats_dense, query_attn_masks
        
    @torch.no_grad()
    def get_box_prompt(
        self, img_features, img_pos, boxes, key_fid=None, key_fid_original=None, 
        is_train=False, enable_dense_prompt=True,
    ):
        """

        Args:
            img_features: C x H/16 x W/16
            img_pos: C x H/16 x W/16
            boxes: Q x 4
            is_train: bool

        Returns:
            point_coords: Q x 2
            query_pe: Q x L x T x C
            query_feats: Q x L x T x C, directly repeated features in temporal direction
            query_attn_masks: T x 1 x Q x (HW/16**2)

        """
        key_fid = self.key_fid if key_fid is None else key_fid
        if key_fid_original is None:
            key_fid_original = key_fid

        h_img, w_img = img_features.shape[-2:]
        device = img_features.device
        _num_insts = boxes.shape[0]

        assert boxes.dim() == 2
        valid = (box_xyxy_to_cxcywh(boxes)[..., 2:] > 0).all(-1)

        point_coords = self.select_points_from_box_mask(h_img, w_img, boxes=boxes)
        # generate positional embeddings for points: Q, C -> Q, T, C
        input_size = (self.num_frames, h_img*self.img_feats_scale, w_img*self.img_feats_scale)
        if self.position_embedding_sin3d_type == "FixedT":
            query_pe = self.pe_layer.forward_points_with_size(input_size, point_coords)
            query_pe = query_pe[key_fid].unsqueeze(1).repeat(1,self.num_frames,1)
        else:
            query_pe = self.pe_layer.forward_points_with_size(input_size, point_coords, key_fid_original)
            query_pe = query_pe.transpose(0, 1)

        # generate noised boxes during training
        if is_train:  
            boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
            boxes_cxcy, boxes_wh = boxes_cxcywh[:, :2], boxes_cxcywh[:, 2:]

            # only add noise on medium and large objects
            # width and height in [0.75*w, 1.25*w] x [0.75*h, 1.25*w]
            boxes_wh_noised = torch.clamp(
                boxes_wh + 0.1 * boxes_wh * (2 * torch.rand((_num_insts, 2), device=device) - 1),
                min=0, max=1
            )
            # center coords in [cx-0.25*w, cx+0.25*w] x [cy-0.25*h, cy+0.25*h]
            boxes_cxcy_noised = torch.clamp(
                boxes_cxcy + 0.1 * boxes_wh * (2 * torch.rand((_num_insts, 2), device=device) - 1),
                min=0, max=1
            )
            boxes_noised = torch.cat([
                boxes_cxcy_noised - 0.5 * boxes_wh_noised, boxes_cxcy_noised + 0.5 * boxes_wh_noised
            ], dim=-1)
            noised = (boxes_wh.prod(-1) > 0.09).unsqueeze(-1).float()
            boxes = boxes * (1 - noised) + boxes_noised * noised

        box_masks = convert_box_to_mask(boxes, img_features.shape[-2], img_features.shape[-1])
        query_feats = (box_masks[:, None] * img_features[None]).flatten(-2).sum(-1) / \
                       box_masks[:, None].flatten(-2).sum(-1).clamp(min=1)
        blank_boxes = box_masks.flatten(-2).sum(-1) == 0
        if blank_boxes.any():
            point_feats = point_sample(
                img_features.unsqueeze(0),
                point_coords[blank_boxes].unsqueeze(0),
                align_corners=False
            ).squeeze(0).t()  # Q_blank, C
            query_feats[blank_boxes] = point_feats
        query_feats = query_feats[:, None].repeat(1, self.num_frames, 1)

        query_attn_masks = torch.zeros(
            (self.num_frames, 1, box_masks.shape[0], h_img*w_img)
        ).bool().to(device)
        box_attn_masks = convert_box_to_mask(boxes, h_img, w_img)
        query_attn_masks[key_fid, 0] = torch.logical_not(box_attn_masks.flatten(-2))

        query_feats_dense = query_feats[:, None]
        query_pe_dense = query_pe[:, None]
        if enable_dense_prompt:
            query_feats_dense, query_pe_dense = self.get_dense_features(
                img_features, img_pos, box_masks, query_pe, query_feats, is_train=is_train
            )
        
        if (~valid).any():  # non-appeared objects in the key frame
            query_pe_dense = query_pe_dense * valid.view(-1,1,1,1)
            query_feats_dense = query_feats_dense * valid.view(-1,1,1,1)
            query_attn_masks[:, :, ~valid] = False

        return point_coords, query_pe_dense, query_feats_dense, query_attn_masks
    
    @torch.no_grad()
    def select_points_from_box_mask(
        self, h_img, w_img, boxes=None, masks=None, is_train=False, mask_thresh=0.75, num_points=1
    ):
        """

        Args:
            boxes: Q x 4
            masks: Q x H x W
            is_train: bool
            mask_thresh: mask threshold
            num_points: number of selected points

        Returns:
            point_coords: Q x 2
        """
        assert (boxes is not None) or (masks is not None)

        if masks is not None:
            device = masks.device
            _num_insts, h, w = masks.shape

            masks = masks.float()
            assert (h_img * self.img_feats_scale == h) and (w_img * self.img_feats_scale == w), \
                f"Input images must have same size with masks: " \
                f"{(h, w), (h_img * self.img_feats_scale, w_img * self.img_feats_scale)}"
            i, j = torch.meshgrid(torch.arange(h), torch.arange(w))
            input_image_coords = (torch.stack([j, i], dim=-1) + 0.5) / torch.as_tensor([w, h]).view(1,1,-1)
            if boxes is None:
                boxes_wo_normalized = convert_mask_to_box(masks > mask_thresh)
                boxes = boxes_wo_normalized / torch.as_tensor([w, h, w, h]).view(1,1,-1)

            boxes_cxcywh = box_xyxy_to_cxcywh(boxes)

            point_coords = []
            if is_train:
                masks_binary = masks.flatten(-2, -1) >= mask_thresh  # Q, HW
                input_image_coords = input_image_coords[:h, :w].flatten(0, 1).to(device)  # HW, 2

                slt_idxs = torch.randperm(h * w).to(device).repeat(num_points)
                slt_idxs = slt_idxs[:_num_insts * num_points].reshape(_num_insts, -1)
                slt_idxs = torch.fmod(slt_idxs, masks_binary.sum(-1).reshape(-1, 1))  

                # randomly select a pixel in the inner mask
                point_coords = [
                    input_image_coords[m_binary][idxs]
                    for idxs, m_binary in enumerate(slt_idxs, masks_binary)
                ]
            else:
                mask_thresh = masks.flatten(1).max(1)[0].clamp(max=mask_thresh).reshape(-1, 1)
                masks_binary = masks.flatten(-2, -1) >= mask_thresh  # Q, HW
                input_image_coords = input_image_coords[:h, :w].flatten(0, 1).to(device)  # HW, 2

                # give a priority to the central pixels
                rel_pos_ctr = torch.abs(input_image_coords[None] - boxes_cxcywh[:, None, :2])  # Q, HW, 2
                in_box_ctr = (rel_pos_ctr < 0.25 * boxes_cxcywh[:, None, 2:]).all(-1)  # Q, HW
                in_mask_ctr = in_box_ctr & masks_binary
                for i, in_ctr in enumerate(in_mask_ctr):
                    if in_ctr.any():
                        idxs = torch.randperm(in_ctr.sum()).repeat(num_points)[:num_points]
                        point_coords.append(input_image_coords[in_ctr][idxs])
                    else:
                        points_high_conf = masks[i].flatten() >= min(0.95, masks[i].max())
                        idxs = torch.randperm(points_high_conf.sum()).repeat(num_points)[:num_points]
                        point_coords.append(input_image_coords[points_high_conf][idxs])

            point_coords = torch.stack(point_coords)  # Q, num_points, 2
            assert (point_coords <= 1).all(), "Point coordinates should be smaller than 1"

        else:
            device = boxes.device
            boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
            boxes_cxcy, boxes_wh = boxes_cxcywh[:, :2], boxes_cxcywh[:, 2:]
            boxes_cxcy = boxes_cxcy[:, None].repeat(1, num_points, 1)
            boxes_wh = boxes_wh[:, None].repeat(1, num_points, 1)
            
            offsets = torch.rand(boxes_wh.shape).to(device) * 2 - 1  # [-1, 1]

            # coords in [cx-0.25*w, cx+0.25w] x [cy-0.25*h, cy+0.25*h]
            point_coords = boxes_cxcy + offsets * 0.25 * boxes_wh  # Q, num_points, 2

        return point_coords[:, 0] if num_points == 1 else point_coords
    
    @torch.no_grad()
    def get_dense_features(
        self, img_features, img_pos, masks_binary, query_pe, query_feats, prompt_type="masks", is_train=True
    ):
        """
        Args:
            img_features: C x H_feat x W_feat
                 img_pos: C x H_feat x W_feat
            masks_binary: Q x H_feat x W_feat
                query_pe: Q x C
             query_feats: Q x C
             prompt_type: "points", "boxes" or "masks"

        Returns:
            query_feats_dense: Q x L x T x C
               query_pe_dense: Q x L x T x C

        """
        h_img, w_img = img_features.shape[-2:]

        assert img_features.shape[-2:] == masks_binary.shape[-2:], \
            'Mismatch shape {img_features.shape[-2:]}  and {masks_binary.shape[-2:]}'
        img_features = img_features.flatten(-2).t()
        img_pos = img_pos.flatten(-2).t()
        boxes = convert_mask_to_box(masks_binary)

        query_feats_dense, query_pe_dense = [], []
        for i, (mask_i, box_i) in enumerate(zip(masks_binary, boxes)):
            feat_idx = torch.nonzero(mask_i.flatten()).reshape(-1)
            if len(feat_idx) == 0:
                query_feats_dense.append(query_feats[i, 0].reshape(1, -1).repeat(self.num_dense_points, 1))
                query_pe_dense.append(query_pe[i, 0].reshape(1, -1).repeat(self.num_dense_points, 1))

            else:
                if len(feat_idx) < self.num_dense_points:
                    feat_idx = feat_idx.repeat(int(self.num_dense_points / len(feat_idx)) + 1)[:self.num_dense_points]
                else:
                    feat_idx = feat_idx[torch.randperm(len(feat_idx))[:self.num_dense_points]]
                    
                    if prompt_type == "masks" and is_train:
                        # must includes four extreme points
                        left_edge, top_edge, right_edge, bottom_edge = box_i
                        feat_idx[0] = torch.nonzero(mask_i[:, left_edge])[0] * w_img + left_edge
                        feat_idx[1] = top_edge * w_img + torch.nonzero(mask_i[top_edge, :])[0]
                        feat_idx[2] = torch.nonzero(mask_i[:, right_edge])[0] * w_img + right_edge
                        feat_idx[3] = bottom_edge * w_img + torch.nonzero(mask_i[bottom_edge, :])[0]
                        
                query_feats_dense.append(img_features[feat_idx])
                query_pe_dense.append(img_pos[feat_idx])

        query_feats_dense = torch.stack(query_feats_dense)[:, :, None].repeat(1, 1, self.num_frames, 1)  # Q, R, T, C
        query_pe_dense = torch.stack(query_pe_dense)[:, :, None].repeat(1, 1, self.num_frames, 1)  # Q, R, T, C
        
        return query_feats_dense, query_pe_dense

class VisualPromptSampler:
    def __init__(
            self,
            pretrain_img_size=1024,
            hidden_dim=256,
            num_heads=8,
            num_frames=1,
            num_dense_points=32,
            position_embedding_sin3d_type="FixedT",
            clip_stride=1,
    ):
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.key_fid = int((num_frames - 1)/2)
        self.num_dense_points = num_dense_points  
        self.clip_stride = clip_stride

        self.visual_prompt_encoder = VisualPromptEncoder(
            pretrain_img_size=pretrain_img_size,
            hidden_dim=hidden_dim,
            num_frames=num_frames,
            num_dense_points=num_dense_points,
            position_embedding_sin3d_type=position_embedding_sin3d_type,
        )

        self.prompt_feature_level_index = -1  # 1/8 resolution of input images 

    @torch.no_grad()
    def process_per_batch(
        self, img_emb_list, pos_emb_list, img_size_list, targets, training=True, 
        prompt_type="masks", use_all_prev_frames=False
    ):
        """
         img_emb_list: [H_lW_lxNTxC], l=1,2,3,4, multi-scale features after pixel encoder
         pos_emb_list: [H_lW_lxNTxC], l=1,2,3,4, positioanl embeddings of img emb
         img_size_list: [(H_0, W_0), ...], multi-scale feature sizes
        """
        if not training:
            return self.process_per_batch_inference(
                img_emb_list, pos_emb_list, img_size_list, targets, prompt_type
            )

        img_emb = img_emb_list[self.prompt_feature_level_index]
        pos_emb = pos_emb_list[self.prompt_feature_level_index]
        img_size = img_size_list[self.prompt_feature_level_index]

        img_emb = rearrange(img_emb, '(H W) (N T) C -> N T C H W', H=img_size[0], W=img_size[1], N=len(targets))
        pos_emb = rearrange(pos_emb, '(H W) (N T) C -> N T C H W', H=img_size[0], W=img_size[1], N=len(targets))

        prompt_pe_dense = []
        prompt_feats_dense = []
        prompt_attn_masks = []
        for img_emb_per_video, pos_emb_per_video, targets_per_video in zip(img_emb, pos_emb, targets):
            prompt_outs = self.process_per_video(
                img_emb_per_video, pos_emb_per_video, targets_per_video, use_all_prev_frames
            )
            prompt_pe_dense.append(prompt_outs[0])
            prompt_feats_dense.append(prompt_outs[1])  # num_gt_instsxRxTxC
            prompt_attn_masks.append(prompt_outs[2])
        
        if None in prompt_feats_dense:
            return None, None, None

        prompt_pe_dense = torch.stack(prompt_pe_dense, dim=-3).flatten(-3, -2)  # num_gt_instsxRxNTxC
        prompt_feats_dense = torch.stack(prompt_feats_dense, dim=-3).flatten(-3, -2) # num_gt_instsxRxNTxC
        prompt_attn_masks = torch.stack(prompt_attn_masks, dim=0).flatten(0,1).repeat(1,self.num_heads,1,1).flatten(0, 1)   # (NTh)xnum_gt_instsxHW 

        return prompt_pe_dense, prompt_feats_dense, prompt_attn_masks
    
    @torch.no_grad()
    def process_per_video(
        self, img_emb_per_video, pos_emb_per_video, targets_per_video, use_all_prev_frames=False
    ):
        device = img_emb_per_video.device
        num_gt_insts, num_frames = targets_per_video['masks'].shape[:2]
        if num_gt_insts == 0:
            num_max_insts = targets_per_video['num_max_instances'] if 'num_max_instances' in targets_per_video else num_gt_insts
            targets_per_video["prompt_obj_ids"] = torch.ones(num_max_insts, device=device) * -1
            return None, None, None,

        if not use_all_prev_frames:
            # intra-clip prompt propogation from a key frame to reference frames
            return self.process_per_frame(
                img_emb_per_video, pos_emb_per_video, targets_per_video
            )
        else:
            num_key_frmaes = max(torch.randperm(self.num_frames)[0], int(self.num_frames/2)+1)
            key_fid_list = (torch.randperm(self.num_frames)[:num_key_frmaes]).sort()[0]
            # inter-clip prompt propogation from the previous frames to current frame
            prompt_pe_dense, prompt_feats_dense, prompt_attn_masks = [], [], []
            for key_fid in key_fid_list:
                prompt_tuple = self.process_per_frame(
                    img_emb_per_video, pos_emb_per_video, targets_per_video, key_fid
                )
                if prompt_tuple[0] is not None:
                    prompt_pe_dense.append(prompt_tuple[0][:, :, key_fid])
                    prompt_feats_dense.append(prompt_tuple[1][:, :, key_fid])
                    prompt_attn_masks.append(prompt_tuple[2][key_fid])

            if len(prompt_pe_dense):
                # inter-clip visual prompts: num_insts x T_key*R x T x C
                prompt_pe_dense = torch.cat(prompt_pe_dense, dim=1)[:,:,None].repeat(1,1,self.num_frames,1)
                prompt_feats_dense = torch.cat(prompt_feats_dense, dim=1)[:,:,None].repeat(1,1,self.num_frames,1)
                prompt_attn_masks = (torch.stack(prompt_attn_masks).float().sum(0) == len(prompt_attn_masks))[None].repeat(self.num_frames,1,1,1)
                return prompt_pe_dense, prompt_feats_dense, prompt_attn_masks
            else:
                return None, None, None

    @torch.no_grad()
    def process_per_frame(
        self, img_emb_per_video, pos_emb_per_video, targets_per_video, key_fid=None
    ):
        """
        sparse prompts include points and similar images, and dense prompts include box and mask of objects.
        The ratio between sparse : dense = 0.5 : 0.5.
        Args:
            img_emb_per_video: TxCxHxW, features with shape
            pos_emb_per_video: TxCxHxW,
            targets_per_video: a dict includes the ground-truth objects of per image/video

        Returns:
                 prompt_coords: point coordinates of prompt
               prompt_pe_dense: point pos emb of prompt, Q x L x T x C
            prompt_feats_dense: point feats of prompt,   Q x L x T x C
             prompt_attn_masks: attention mask of dense prompt, box or mask, T x 1 x Q x (HW/16**2)
        """
        key_fid = torch.randperm(self.num_frames)[0] if key_fid is None else key_fid
        if "frame_indices" in targets_per_video:
            key_fid_original = targets_per_video["frame_indices"][key_fid]
        else:
            key_fid_original = key_fid

        device = img_emb_per_video.device
        x_key = img_emb_per_video[key_fid]  # C x H x W
        x_pos_key = pos_emb_per_video[key_fid]  # C x H x W
        gt_boxes = targets_per_video["boxes"].to(device)  # num_gt_insts x T x 4
        gt_masks = targets_per_video['masks'].to(device)  # num_gt_insts x T x H x W
        # the max number of instances in a batch for parallel 
        num_gt_insts = len(gt_masks) 
        num_max_insts = targets_per_video['num_max_instances'] if 'num_max_instances' in targets_per_video else num_gt_insts 
        H_gt, W_gt = gt_masks.shape[-2:]

        # assign a prompt type for each object, including point, box and mask
        flag_prompt_types = torch.ones(num_gt_insts, device=device) 
        random_prompt_types = torch.rand(num_gt_insts, device=device)
        flag_prompt_types[random_prompt_types <= 0.25] = 0  # is_point_prompts
        flag_prompt_types[(random_prompt_types > 0.25) & (random_prompt_types <= 0.5)] = 1  # is_box_prompts
        flag_prompt_types[random_prompt_types > 0.5] = 2  # is_mask_prompts

        is_point_prompts = flag_prompt_types == 0
        is_box_prompts = flag_prompt_types == 1
        is_mask_prompts = flag_prompt_types == 2
        gt_idxs = torch.arange(num_gt_insts, device=device)

        pbm_gt_idxs = []
        pbm_prompt_tuples = [[], [], [], []]
        if is_mask_prompts.any():
            gt_idxs_mask = gt_idxs[is_mask_prompts]
            dense_masks = gt_masks[gt_idxs_mask, key_fid]
            dense_boxes = gt_boxes[gt_idxs_mask, key_fid]
            obj_prompt_tuple = self.visual_prompt_encoder.get_mask_prompt(
                x_key, x_pos_key, masks=dense_masks, boxes=dense_boxes, is_train=True,
                key_fid=key_fid, key_fid_original=key_fid_original
            )
            for k, v in enumerate(obj_prompt_tuple):
                pbm_prompt_tuples[k].append(v)

            pbm_gt_idxs.append(gt_idxs_mask)

        if is_point_prompts.any():
            gt_idxs_point = gt_idxs[is_point_prompts]
            point_idxs = torch.randperm(H_gt*W_gt, device=device)[:is_point_prompts.sum()]
            for i_, (gt_idx, point_idx) in enumerate(zip(gt_idxs_point, point_idxs)):
                # randomly select a point in the inner mask of the target object as prompt
                point_idx_in_masks = torch.nonzero(gt_masks[gt_idx, key_fid].flatten(-2).gt(0.5)).reshape(-1)
                if len(point_idx_in_masks) > 0:
                    point_idxs[i_] = point_idx_in_masks[point_idx % len(point_idx_in_masks)]
                else:
                    gt_idxs_point[i_] = -1

            point_x = (point_idxs % W_gt + 0.5) / W_gt
            point_y = ((point_idxs / W_gt).floor() + 0.5) / H_gt
            point_coords = torch.stack([point_x, point_y], dim=-1)  # q_point, 2
            # assign -1 for unappeared objects
            point_coords[gt_idxs_point == -1] = -1
            obj_prompt_tuple = self.visual_prompt_encoder.get_point_prompt(
                x_key, x_pos_key, point_coords, is_train=True,
                key_fid=key_fid, key_fid_original=key_fid_original
            )
            for k, v in enumerate(obj_prompt_tuple):
                pbm_prompt_tuples[k].append(v)

            pbm_gt_idxs.append(gt_idxs_point)

        if is_box_prompts.any():
            gt_idxs_box = gt_idxs[is_box_prompts]
            dense_boxes = gt_boxes[gt_idxs_box, key_fid]
            obj_prompt_tuple = self.visual_prompt_encoder.get_box_prompt(
                x_key, x_pos_key, dense_boxes, is_train=True,
                key_fid=key_fid, key_fid_original=key_fid_original
            )
            for k, v in enumerate(obj_prompt_tuple):
                pbm_prompt_tuples[k].append(v)

            pbm_gt_idxs.append(gt_idxs_box)

        prompt_coords = torch.cat(pbm_prompt_tuples[0])
        prompt_coords = repeat(prompt_coords, 'Q C -> Q T C', T=self.num_frames)
        prompt_pe_dense = torch.cat(pbm_prompt_tuples[1])     # num_gt_instsxRxTxC
        prompt_feats_dense = torch.cat(pbm_prompt_tuples[2])  # num_gt_instsxRxTxC
        prompt_attn_masks = torch.cat(pbm_prompt_tuples[3], dim=-2)  

        # resort the order of visual prompts to the original order of ground-truth obj ids
        pbm_gt_idxs = torch.cat(pbm_gt_idxs).sort()[1]
        prompt_pe_dense = prompt_pe_dense[pbm_gt_idxs]
        promt_feats_dense = prompt_feats_dense[pbm_gt_idxs]
        prompt_attn_masks = prompt_attn_masks[..., pbm_gt_idxs, :]
        targets_per_video["prompt_obj_ids"] = gt_idxs.long()  # corresponding to ground-truth obj ids

        # for parallel in a batch
        prompt_pe_dense = prompt_pe_dense[:num_max_insts]   
        prompt_feats_dense = prompt_feats_dense[:num_max_insts]
        prompt_attn_masks = prompt_attn_masks[:, :, :num_max_insts]
        targets_per_video["prompt_obj_ids"] = targets_per_video["prompt_obj_ids"][:num_max_insts]

        # for parallel in a batch
        if prompt_feats_dense.shape[0] < num_max_insts:
            padding_obj_ids = torch.arange(prompt_feats_dense.shape[0]).repeat(num_max_insts)
            padding_obj_ids = padding_obj_ids[:num_max_insts-prompt_feats_dense.shape[0]]
            prompt_pe_dense = torch.cat([prompt_pe_dense, prompt_pe_dense[padding_obj_ids]])
            prompt_feats_dense = torch.cat([prompt_feats_dense, prompt_feats_dense[padding_obj_ids]])
            prompt_attn_masks = torch.cat([prompt_attn_masks, prompt_attn_masks[..., padding_obj_ids, :]], dim=2)
            targets_per_video["prompt_obj_ids"] = torch.cat(
                [targets_per_video["prompt_obj_ids"], targets_per_video["prompt_obj_ids"][padding_obj_ids]]
            )
        return prompt_pe_dense, prompt_feats_dense, prompt_attn_masks
    
    @torch.no_grad()
    def process_per_batch_inference(
        self, img_emb_list, pos_emb_list, img_size_list, targets, prompt_type="masks"
    ):
        """
         img_emb_list: [H_lW_lxNTxC], l=1,2,3,4, multi-scale features after pixel encoder
         pos_emb_list: [H_lW_lxNTxC], l=1,2,3,4, positioanl embeddings of img emb
         img_size_list: [(H_0, W_0), ...], multi-scale feature sizes
         prompt_type: the type of prompt annotation, which should be in "points", "boxes", "masks"
        """
        assert len(targets) == 1, 'Only support batch size = 1 now'

        img_emb = img_emb_list[self.prompt_feature_level_index]
        pos_emb = pos_emb_list[self.prompt_feature_level_index]
        img_size = img_size_list[self.prompt_feature_level_index]

        img_emb = rearrange(img_emb, '(H W) (N T) C -> N T C H W', H=img_size[0], W=img_size[1], N=len(targets))
        pos_emb = rearrange(pos_emb, '(H W) (N T) C -> N T C H W', H=img_size[0], W=img_size[1], N=len(targets))

        prompt_pe_dense = []
        prompt_feats_dense = []
        prompt_attn_masks = []
        for img_emb_per_video, pos_emb_per_video, targets_per_video in zip(img_emb, pos_emb, targets):
            targets_per_video["img_emb_per_video"] = img_emb_per_video
            targets_per_video["pos_emb_per_video"] = pos_emb_per_video
            if 'masks' not in targets_per_video or targets_per_video['masks'].nelement() == 0:
                return None, None, None

            prompt_outs = self.process_per_video_inference(
                img_emb_per_video, pos_emb_per_video, targets_per_video, prompt_type
            )
            prompt_pe_dense.append(prompt_outs[0])
            prompt_feats_dense.append(prompt_outs[1])  # num_gt_instsxRxTxC
            prompt_attn_masks.append(prompt_outs[2])
        
        if len(prompt_feats_dense) == 0 or None in prompt_feats_dense:
            return None, None, None

        prompt_pe_dense = torch.stack(prompt_pe_dense, dim=-3).flatten(-3, -2)        # num_gt_instsxRxNTxC
        prompt_feats_dense = torch.stack(prompt_feats_dense, dim=-3).flatten(-3, -2)  # num_gt_instsxRxNTxC
        prompt_attn_masks = torch.stack(prompt_attn_masks, dim=0).flatten(0,1).repeat(1,self.num_heads,1,1).flatten(0, 1)   # (NTh)xnum_gt_instsxHW 
        
        isblank = (prompt_feats_dense == 0).all(-1)
        prompt_feats_mean = (prompt_feats_dense * isblank.unsqueeze(-1)).flatten(1,2).sum(1) / isblank.flatten(1,2).sum(1).unsqueeze(-1)
        prompt_feats_mean = prompt_feats_mean[:,None,None].repeat(1,prompt_feats_dense.shape[1], prompt_feats_dense.shape[2], 1)
        prompt_feats_dense[isblank] = prompt_feats_mean[isblank].clone().detach()

        return prompt_pe_dense, prompt_feats_dense, prompt_attn_masks
    
    @torch.no_grad()
    def process_per_video_inference(
        self, img_emb_per_video, pos_emb_per_video, targets_per_video, prompt_type="masks"
    ):
        """
        sparse prompts include points and similar images, and dense prompts include box and mask of objects
        Args:
            img_emb_per_video: TxCxHxW, features with shape
            pos_emb_per_video: TxCxHxW,
            targets_per_video: a dict includes the ground-truth objects of per image/video
            prompt_type: the type of prompt annotation, which should be "points", "boxes", "masks"

        Returns:
            prompt_coords: point coordinates of prompt
            prompt_pe_dense: point pos emb of prompt
            prompt_feats_dense: point feats of prompt
            prompt_attn_masks: attention mask of dense prompt, box or mask
        """
        device = img_emb_per_video.device
        num_frames = img_emb_per_video.shape[0]

        # the frame index which is the first frame of the processing video clip
        first_frame_idx = targets_per_video["first_frame_idx"]
        frame_indices = targets_per_video["frame_indices"]
        is_first_clip = first_frame_idx == 0

        if not is_first_clip:
            self.zero_pad_prompt(targets_per_video)
            self.process_per_video_inference_prev_frame(
                targets_per_video, prompt_type="masks"
            )

        gt_boxes = targets_per_video["boxes"][:, -num_frames:].to(device)  # num_gt_insts x T x 4
        gt_masks = targets_per_video['masks'][:, -num_frames:].to(device)  # num_gt_insts x T x H x W
        num_gt_insts, _, H_gt, W_gt = gt_masks.shape

        update_frame_ids = torch.nonzero(gt_masks.gt(0.).sum(dim=(0,2,3)) > 0).reshape(-1)        
        for key_fid in update_frame_ids:
            key_fid_original = frame_indices[key_fid]
            x_key = img_emb_per_video[key_fid]      # C x H x W
            x_pos_key = pos_emb_per_video[key_fid]  # C x H x W

            assert prompt_type in {"points", "boxes", "masks"}
            if prompt_type == "points":
                point_coords = []
                # randomly select a point in the inner mask of the target object as prompt
                point_idxs = torch.randperm(H_gt*W_gt, device=device)[:has_appeared.sum()]
                for i_, point_idx in enumerate(point_idxs):
                    point_idx_in_masks = torch.nonzero(gt_masks[i_, key_fid].flatten(-2).gt(0.5)).reshape(-1)
                    if len(point_idx_in_masks) > 0:
                        point_idx = point_idx_in_masks[point_idx % len(point_idx_in_masks)]
                        point_x = (point_idx % W_gt + 0.5) / W_gt
                        point_y = ((point_idx / W_gt).floor() + 0.5) / H_gt
                        point_coords.append([point_x, point_y])
                    else:
                        w, h = gt_boxes[i_, key_fid, 2:] - gt_boxes[i_, key_fid, :2]
                        if w > 0 and h > 0:
                            cx, cy = 0.5 * (gt_boxes[i_, key_fid, :2] + gt_boxes[i_, key_fid, 2:])
                            point_idx = cy * W_gt + cx
                            point_x = (point_idx % W_gt + 0.5) / W_gt
                            point_y = ((point_idx / W_gt).floor() + 0.5) / H_gt
                            point_coords.append([point_x, point_y])
                        else:
                            point_coords.append([-1, -1])  # non-used points

                point_coords = torch.as_tensor(point_coords)  # q_point, 2
                obj_prompt_tuple = self.visual_prompt_encoder.get_point_prompt(
                    x_key, x_pos_key, point_coords, is_train=False,
                    key_fid=key_fid, key_fid_original=key_fid_original
                )

            if prompt_type == "boxes":
                dense_boxes = gt_boxes[:, key_fid]
                obj_prompt_tuple = self.visual_prompt_encoder.get_box_prompt(
                    x_key, x_pos_key, dense_boxes, is_train=False,
                    key_fid=key_fid, key_fid_original=key_fid_original
                )

            if prompt_type == "masks":
                dense_masks = gt_masks[:, key_fid]
                dense_boxes = gt_boxes[:, key_fid]
                obj_prompt_tuple = self.visual_prompt_encoder.get_mask_prompt(
                    x_key, x_pos_key, masks=dense_masks, boxes=dense_boxes, is_train=False,
                    key_fid=key_fid, key_fid_original=key_fid_original
                )

            prompt_coords = repeat(obj_prompt_tuple[0], 'Q C -> Q T C', T=num_frames)
            prompt_pe_dense = obj_prompt_tuple[1]    # num_gt_instsxRxTxC
            prompt_feats_dense = obj_prompt_tuple[2]
            prompt_attn_masks = obj_prompt_tuple[3]
            targets_per_video["prompt_obj_ids"] = targets_per_video["ids"]  # corresponding to ground-truth obj ids
            if is_first_clip:
                targets_per_video["prompt_obj_ids"] = targets_per_video["ids"]
                targets_per_video["prompt_pe"] = prompt_pe_dense
                targets_per_video["prompt_feats"] = prompt_feats_dense
                targets_per_video["prompt_attn_masks"] = prompt_attn_masks
            else:
                s_idx = targets_per_video["prompt_feats"].shape[-2] - num_frames + key_fid
                valid = gt_masks[:, key_fid].flatten(1).sum(1) > 0
                targets_per_video["prompt_pe"][valid,:, s_idx:] = prompt_pe_dense[valid,:,key_fid:]
                targets_per_video["prompt_feats"][valid,:, s_idx:] = prompt_feats_dense[valid,:,key_fid:]
                targets_per_video["prompt_attn_masks"][s_idx:] = prompt_attn_masks[key_fid:]
        
        if "prompt_pe" not in targets_per_video:
            return None, None, None
        else:
            prompt_pe_dense = targets_per_video["prompt_pe"][:,:,-num_frames:] 
            prompt_feats_dense = targets_per_video["prompt_feats"][:,:,-num_frames:] 
            prompt_attn_masks = targets_per_video["prompt_attn_masks"][-num_frames:]
        
            return prompt_pe_dense, prompt_feats_dense, prompt_attn_masks

    @torch.no_grad()
    def process_per_video_inference_prev_frame(
        self, targets_per_video, prompt_type="masks"
    ):
        """
        sparse prompts include points and similar images, and dense prompts include box and mask of objects
        Args:
            img_emb_per_video: 1xCxHxW, features of the previous frame
            pos_emb_per_video: 1xCxHxW, pos of the previous frame
            targets_per_video: a dict includes the ground-truth objects of per image/video
            prompt_type: the type of prompt annotation, which should be "points", "boxes", "masks"

        Returns:
            prompt_coords: point coordinates of prompt
            prompt_pe_dense: point pos emb of prompt
            prompt_feats_dense: point feats of prompt
            prompt_attn_masks: attention mask of dense prompt, box or mask
        """
        device = targets_per_video["img_emb_per_video"].device
        num_gt_insts = targets_per_video['masks'].shape[0]
        num_frames = targets_per_video["img_emb_per_video"].shape[0]

        # the index of the last frame: t-1
        prev_frame_idx = max(0, targets_per_video["first_frame_idx"]-1) 
        has_appeared = (targets_per_video["first_appear_frame_idxs"] <= prev_frame_idx) & \
                       (targets_per_video["first_appear_frame_idxs"] != -1)
        update_prev_frame = (self.num_frames == 1) or ("prompt_feats" not in targets_per_video)
        if has_appeared.sum() == 0 and not update_prev_frame:
            return 

        for key_fid in range(self.clip_stride):
            gt_boxes = targets_per_video["boxes"][:, -(num_frames+self.clip_stride)+key_fid].to(device)  # num_gt_insts x 4
            gt_masks = targets_per_video['masks'][:, -(num_frames+self.clip_stride)+key_fid].to(device)  # num_gt_insts x H x W
            H_gt, W_gt = gt_masks.shape[-2:]

            gt_boxes = gt_boxes[has_appeared]
            gt_masks = gt_masks[has_appeared]

            key_fid_original = targets_per_video["frame_indices"][0] - 1
            x_key = targets_per_video["img_emb_per_video"][key_fid]      # C x H x W
            x_pos_key = targets_per_video["pos_emb_per_video"][key_fid]  # C x H x W

            assert prompt_type in {"points", "boxes", "masks"}
            if prompt_type == "points":
                point_coords = []
                # randomly select a point in the inner mask of the target object as prompt
                point_idxs = torch.randperm(H_gt*W_gt, device=device)[:has_appeared.sum()]
                for i_, point_idx in enumerate(point_idxs):
                    point_idx_in_masks = torch.nonzero(gt_masks[i_].flatten(-2).gt(0.5)).reshape(-1)
                    if len(point_idx_in_masks) > 0:
                        point_idx = point_idx_in_masks[point_idx % len(point_idx_in_masks)]
                        point_x = (point_idx % W_gt + 0.5) / W_gt
                        point_y = ((point_idx / W_gt).floor() + 0.5) / H_gt
                        point_coords.append([point_x, point_y])
                    else:
                        w, h = gt_boxes[i_, 2:] - gt_boxes[i_, :2]
                        if w > 0 and h > 0:
                            cx, cy = 0.5 * (gt_boxes[i_, :2] + gt_boxes[i_, 2:])
                            point_idx = cy * W_gt + cx
                            point_x = (point_idx % W_gt + 0.5) / W_gt
                            point_y = ((point_idx / W_gt).floor() + 0.5) / H_gt
                            point_coords.append([point_x, point_y])
                        else:
                            point_coords.append([-1, -1])  # non-used points

                point_coords = torch.as_tensor(point_coords)  # q_point, 2
                obj_prompt_tuple = self.visual_prompt_encoder.get_point_prompt(
                    x_key, x_pos_key, point_coords, is_train=False,
                    key_fid=key_fid, key_fid_original=key_fid_original
                )

            if prompt_type == "boxes":
                obj_prompt_tuple = self.visual_prompt_encoder.get_box_prompt(
                    x_key, x_pos_key, gt_boxes, is_train=False,
                    key_fid=key_fid, key_fid_original=key_fid_original
                )

            if prompt_type == "masks":
                obj_prompt_tuple = self.visual_prompt_encoder.get_mask_prompt(
                    x_key, x_pos_key, masks=gt_masks, boxes=gt_boxes, is_train=False,
                    key_fid=key_fid, key_fid_original=key_fid_original, 
                )

            prompt_coords = repeat(obj_prompt_tuple[0], 'Q C -> Q T C', T=num_frames)
            prompt_pe_dense = obj_prompt_tuple[1]  # num_gt_instsxRxTxC
            prompt_feats_dense = obj_prompt_tuple[2]
            prompt_attn_masks = obj_prompt_tuple[3]

            if "prompt_feats" not in targets_per_video:
                _, R, T, C = prompt_pe_dense.shape
                targets_per_video["prompt_pe"] =  torch.zeros([num_gt_insts, R, T+self.clip_stride, C], device=device)
                targets_per_video["prompt_feats"] =  torch.zeros([num_gt_insts, R, T+self.clip_stride, C], device=device)
                targets_per_video["prompt_attn_masks"] = torch.zeros(
                    [prompt_attn_masks.shape[0]+self.clip_stride, prompt_attn_masks.shape[1], num_gt_insts, prompt_attn_masks.shape[-1]], device=device
                ).bool()

            targets_per_video["prompt_pe"][has_appeared,:,-(num_frames+self.clip_stride)+key_fid] = prompt_pe_dense[:,:,key_fid]
            targets_per_video["prompt_feats"][has_appeared,:,-(num_frames+self.clip_stride)+key_fid] = prompt_feats_dense[:,:,key_fid]
            targets_per_video["prompt_attn_masks"][-(num_frames+self.clip_stride)+key_fid, :, has_appeared] = prompt_attn_masks[key_fid]

    @torch.no_grad()
    def zero_pad_prompt(self, targets_per_video):
        if "prompt_feats" not in targets_per_video:
            return 

        # for the newly frame, we set the mean of prompt feats in all previous frames as the initial prompt tokens
        zero_pad_tensor = torch.zeros_like(targets_per_video["prompt_pe"][:,:,-self.clip_stride:])
        targets_per_video["prompt_pe"] = torch.cat([targets_per_video["prompt_pe"], zero_pad_tensor], dim=2)
        targets_per_video["prompt_feats"] = torch.cat([targets_per_video["prompt_feats"], zero_pad_tensor], dim=2)
        targets_per_video["prompt_attn_masks"] = torch.cat(
            [targets_per_video["prompt_attn_masks"], targets_per_video["prompt_attn_masks"][-self.clip_stride:]]
        )
        targets_per_video["prompt_attn_masks"][-self.clip_stride:] = False

def generate_temporal_weights(num_frames, weights=None, enable_softmax=False):
    """
    num_frames: the number of frames in the temporal dimension 
       weights: ...xT, the weights of each frame, it should be a vector with the same length with the number of frames
    """
    temp_w = (torch.arange(1, num_frames+1).float() / num_frames * 10.0).exp()
    if enable_softmax:
        temp_w = temp_w.softmax(-1)

    if weights is not None:
        assert weights.shape[-1] == num_frames, "Inconsistency length between weights and values"
        temp_w = temp_w.to(weights) * weights

    return temp_w / temp_w.sum(-1).unsqueeze(-1).clamp(min=1e-3)
    
    