# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import os
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange, repeat

from timm.models.layers import trunc_normal_

from detectron2.config import configurable
from detectron2.layers import Conv2d

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .position_encoding import PositionEmbeddingSine3D, PositionEmbeddingSine3DArbitraryT
from .transformer_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer, MLP

from univs.modeling.prompt_encoder import VisualPromptEncoder, VisualPromptSampler
from datasets.concept_emb.combined_datasets_category_info import combined_datasets_category_info


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoderUniVS(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        prompt_self_attn_layers: int = -1,
        # video related
        num_frames: int=1,
        clip_class_embed_path: str,
        visual_prompt_sampler,
        num_dense_points: int,
        text_prompt_enable: bool=True,
        prompt_as_queries: bool=True, 
        text_prompt_to_image_enable: bool=True,
        maskdec_self_attn_mask_type: str='sep',
        disable_learnable_queries_sa1b: bool=False,
        position_embedding_sin3d_type: str='FixedT',
        # inference parameters
        num_prev_frames_memory: int=10, 
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_frames = num_frames

        # positional encoding
        N_steps = hidden_dim // 2
        self.position_embedding_sin3d_type = position_embedding_sin3d_type
        if self.position_embedding_sin3d_type == "FixedT":
            self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)
        else:
            assert self.position_embedding_sin3d_type == "ArbitraryT"
            self.pe_layer = PositionEmbeddingSine3DArbitraryT(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.transformer_prompt_self_attention_layers = nn.ModuleList()

        self.prompt_self_attn_layers = self.num_layers if prompt_self_attn_layers < 0 else prompt_self_attn_layers

        for i in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            if i < self.prompt_self_attn_layers:
                self.transformer_prompt_self_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0
                    )
                )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

         # convert video category to text embeddings
        self.clip_cls_text_emb = torch.load(clip_class_embed_path, map_location=self.decoder_norm.weight.device)
        # vis2lang head and lang2vis head
        self.text_emb_dim = self.clip_cls_text_emb.shape[-1]
        self.vis2text_projection = nn.Linear(hidden_dim, self.text_emb_dim)
        self.text_norm = nn.LayerNorm(self.text_emb_dim)
        self.text2vis_projection = nn.Linear(self.text_emb_dim, hidden_dim)
        self.cls_temp = nn.Embedding(1, 1)  # torch.ones([]) * math.log(1 / 0.07))
        self.reid_temp = nn.Embedding(1, 1) # torch.ones([]) * math.log(1 / 0.07))

        self.maskdec_self_attn_mask_type = maskdec_self_attn_mask_type
        # task prompt embeddings
        self.prompt_detection = nn.Embedding(1, hidden_dim)
        self.prompt_sot = nn.Embedding(1, hidden_dim)
        self.prompt_grounding = nn.Embedding(1, hidden_dim)
        # prompt-related parameters
        self.visual_prompt_sampler = visual_prompt_sampler
        self.num_dense_points = num_dense_points
        self.visual_prompt_enable = visual_prompt_sampler is not None
        self.text_prompt_enable = text_prompt_enable
        self.prompt_as_queries = prompt_as_queries
        self.text_prompt_to_image_enable = text_prompt_to_image_enable
        if text_prompt_to_image_enable:
            self.lang2vision_cross_attention_layer = CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    need_weights=True, 
                )
        
        # inference parameters
        self.num_prev_frames_memory = max(num_prev_frames_memory, num_frames+1)
        self.use_visual_prompts_grounding = True

        if self.training:
            self._reset_parameters()
    
    def _reset_parameters(self):
        init_temp_value = torch.ones(1, 1) * math.log(1 / 0.07)
        self.cls_temp.weight.data = init_temp_value
        self.reid_temp.weight.data = init_temp_value

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        visual_prompt_sampler = None
        if cfg.MODEL.UniVS.VISUAL_PROMPT_ENCODER:
            visual_prompt_sampler = VisualPromptSampler(
                pretrain_img_size=cfg.INPUT.LSJ_AUG.IMAGE_SIZE,
                hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
                num_heads=cfg.MODEL.MASK_FORMER.NHEADS, 
                num_frames=cfg.INPUT.SAMPLING_FRAME_NUM, 
                num_dense_points=cfg.MODEL.UniVS.VISUAL_PROMPT_PIXELS_PER_IMAGE,
                position_embedding_sin3d_type=cfg.MODEL.UniVS.POSITION_EMBEDDING_SINE3D,
                clip_stride=cfg.MODEL.BoxVIS.TEST.CLIP_STRIDE,
            )

        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM
        ret["clip_class_embed_path"] = cfg.MODEL.UniVS.CLIP_CLASS_EMBED_PATH
        ret["visual_prompt_sampler"] = visual_prompt_sampler
        ret["num_dense_points"] = cfg.MODEL.UniVS.VISUAL_PROMPT_PIXELS_PER_IMAGE
        ret["text_prompt_enable"] = cfg.MODEL.UniVS.TEXT_PROMPT_ENCODER
        ret["prompt_as_queries"] = cfg.MODEL.UniVS.PROMPT_AS_QUERIES
        ret["text_prompt_to_image_enable"] = cfg.MODEL.UniVS.TEXT_PROMPT_TO_IMAGE_ENABLE
        ret["maskdec_self_attn_mask_type"] = cfg.MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE
        ret["disable_learnable_queries_sa1b"] = cfg.MODEL.UniVS.DISABLE_LEARNABLE_QUERIES_SA1B
        ret["prompt_self_attn_layers"] = cfg.MODEL.UniVS.PROMPT_SELF_ATTN_LAYERS
        ret["position_embedding_sin3d_type"] = cfg.MODEL.UniVS.POSITION_EMBEDDING_SINE3D
        ret["num_prev_frames_memory"] = cfg.MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY

        return ret

    def forward(self, x, mask_features, mask_features_bfe_conv=None, mask=None, targets=None):
        # if not self.training:
        #     self.plot_mask_features(mask_features, targets)

        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)
        mask_features_bfe_conv = mask_features_bfe_conv.view(bs, t, c_m, h_m, w_m)

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask
        
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            if self.position_embedding_sin3d_type == "FixedT":
                pos.append(self.pe_layer(x[i].view(bs, t, -1, size_list[-1][0], size_list[-1][1])).flatten(3))
            else:
                if "frame_indices" in targets[0]:
                    frame_indices = torch.stack([targets_per_video["frame_indices"] for targets_per_video in targets])  # bs, t
                else:
                    frame_indices = torch.arange(t, device=x[i].device)[None].repeat(bs,1)
                pos.append(self.pe_layer(x[i].view(bs, t, -1, size_list[-1][0], size_list[-1][1]), frame_indices).flatten(3))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # NTxCxHW => NxTxCxHW => (TxHW)xNxC
            # _, c, hw = src[-1].shape
            # pos[-1] = pos[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
            # src[-1] = src[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

            # NTxCxHW => HWxNTxC
            pos[-1] = pos[-1].flatten(0, 1).permute(2,0,1)
            src[-1] = src[-1].permute(2,0,1)
        
        # learnable queries: QxNTxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bt, 1)  # QxNTxC
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bt, 1)  # QxNTxC  

        # prompt queires: Q_pxNTxC
        prompt_feats, prompt_pe, prompt_feats_dense, prompt_pe_dense, l2v_attn_weights_list = \
            self.forward_prompt_encoder(src, pos, size_list, targets, t)
        if prompt_feats is not None:
            assert self.prompt_as_queries
            # prompt as queries: QxNTxC
            output = torch.cat([output, prompt_feats])
            prompt_pe = prompt_pe if prompt_pe is not None else prompt_feats
            query_embed = torch.cat([query_embed, prompt_pe])
    
        # prompt cross-attention first (per-frame)
        output = self.forward_transformer_prompt_self_attention_layer(
            0, output, query_embed, prompt_feats_dense, prompt_pe_dense
        )
        query_embed = torch.cat([query_embed[:self.num_queries], output[self.num_queries:]])

        predictions_class = []
        predictions_mask = []
        predictions_embds = []
        predictions_reid = []
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, outputs_reid = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0], 
            task=targets[0]['task'], targets=targets
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_embds.append(rearrange(self.decoder_norm(output), 'Q (B T) C -> B Q T C', T=t))
        predictions_reid.append(outputs_reid)

        num_queries_lp, NT = output.shape[:2]
        dataset_name = targets[0]['dataset_name']
        task = targets[0]['task']
        self_attn_mask = self.generate_self_attn_mask(bs, t, num_queries_lp, output.device, dataset_name, task)
        for i in range(self.num_layers):
            insert_previous_masks = False
            if i < 6 and not self.training and insert_previous_masks:
                # To avoid mask leakage, add predicted masks of previous frames into attn_mask 
                attn_mask = self.prompt_image_attention_mask(
                    attn_mask, size_list[i % self.num_feature_levels], t, targets
                )
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # prompt cross-attention layer
            if i > 0 and i < self.prompt_self_attn_layers:
                output = self.forward_transformer_prompt_self_attention_layer(
                    i, output, query_embed, prompt_feats_dense, prompt_pe_dense
                )

            # cross-attention first (per-frame)
            level_index = i % self.num_feature_levels
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            # spatial-temporal self_attention (QxT queries)
            output = rearrange(output, 'Q (B T) C -> (Q T) B C', T=t)
            query_embed = rearrange(query_embed, 'Q (B T) C -> (Q T) B C', T=t)
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_attn_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            output = rearrange(output, '(Q T) B C -> Q (B T) C', T=t)
            query_embed = rearrange(query_embed, '(Q T) B C -> Q (B T) C', T=t)
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, outputs_reid = self.forward_prediction_heads(
                output, mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], 
                task=targets[0]['task'],
                targets=targets
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_embds.append(rearrange(self.decoder_norm(output), 'Q (B T) C -> B Q T C', T=t))
            predictions_reid.append(outputs_reid)

        assert len(predictions_class) == self.num_layers + 1 

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_reid, predictions_embds
            ),
            'pred_embds': predictions_embds[-1],
            'pred_reid_logits': predictions_reid[-1],
        }
        # if self.training:
        #     out['l2v_attn_weights'] = l2v_attn_weights_list
        
        return out
    
    def forward_transformer_prompt_self_attention_layer(
        self, i, output, query_emb, prompt_feats_dense, prompt_pos_dense
    ):
        """
        Args:
            output:     (Q + K) x NT x C, the query content tokens, Q and K are the numbers of learnable and prompt queries 
            query_emb:  (Q + K) x NT x C, the query pos emb
            prompt_feats_dense: K x L x NT x C, where L is the length of prompt features, K is the number of 
                                classes, expressions and objects for detection, grouding and sot tasks, respectively.
            prompt_pos_dense: K x L x NT x C, pos emb for visual prompt, otherwise None

        return:
            output:     (Q + K) x NT x C, the query content token of prompt
        """
        if output.shape[0] == self.num_queries:
            return output

        output_learn, output_prompt = output[:self.num_queries], output[self.num_queries:]
        query_emb_learn, query_emb_prompt = query_emb[:self.num_queries], query_emb[self.num_queries:]

        # prompt_feats_dense = torch.cat([output_prompt.unsqueeze(1), prompt_feats_dense], dim=1)  # Kx(1+L)xNTxC
        prompt_feats_dense = prompt_feats_dense.transpose(0,1).flatten(1,2)  # (1+L)xKNTxC
        if prompt_pos_dense is not None:
            # prompt_pos_dense = torch.cat([query_emb_prompt.unsqueeze(1), prompt_pos_dense], dim=1)
            prompt_pos_dense = prompt_pos_dense.transpose(0,1).flatten(1,2)
            query_emb_prompt = query_emb_prompt.flatten(0, 1)[None]
        else:
            prompt_pos_dense, query_emb_prompt = None, None
        
        K, NT, _ = output_prompt.shape
        output_prompt = output_prompt.flatten(0, 1)[None]  # 1xKNTxC
        output_prompt = self.transformer_prompt_self_attention_layers[i](
            output_prompt, prompt_feats_dense, 
            pos=prompt_pos_dense, query_pos=query_emb_prompt
        )
        output_prompt = rearrange(output_prompt.squeeze(0), '(K N) C -> K N C', K=K)
        output = torch.cat([output_learn, output_prompt])

        return output

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, task, targets):
        bs, t, c_m, h_m, w_m = mask_features.shape

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # (NT)QC, N is the batch size

        outputs_class = self.vis2text_projection(decoder_output)
        if task != 'grounding':
            CLIP_class = F.normalize(self.clip_cls_text_emb.to(output), p=2, dim=-1).detach()
            outputs_class = F.normalize(outputs_class, p=2, dim=-1)
            outputs_class = torch.einsum('bqc,kc->bqk', outputs_class, CLIP_class)
            outputs_class = rearrange(outputs_class, '(B T) Q C -> B T Q C', T=t).mean(1)
            outputs_class = outputs_class * self.cls_temp.weight.exp()

        else:
            CLIP_exp = torch.stack(
                [targets_per_video['exp_sentence_feats'][:, 0] for targets_per_video in targets]
            ).to(outputs_class).detach()
            outputs_class = rearrange(outputs_class, '(B T) Q C -> B T Q C', T=t).mean(1)
            outputs_class = torch.einsum('bqc,bkc->bqk', outputs_class, CLIP_exp) / outputs_class.shape[-1]
        
        mask_embed = self.mask_embed(decoder_output)  # N'QC
        mask_embed = rearrange(mask_embed, '(B T) Q C -> B T Q C', T=t)
        # randomly sort query embeddings in temporal dimension: [1, 2, 3] -> [2, 3, 1]
        # q2 @ F1_pixel = M1,  q3 @ F2_pixel = M2, q1 @ F3_pixel = M3
        if self.training:
            mask_embed = mask_embed[:, torch.randperm(t)]
        outputs_mask = torch.einsum("btqc,btchw->btqhw", mask_embed, mask_features)
        outputs_mask = outputs_mask.transpose(1, 2)
        b, q, t, _, _ = outputs_mask.shape

        if self.training:
            output = rearrange(self.decoder_norm(output), 'q (b t) c -> (b q t) c', t=t)
            outputs_reid = torch.einsum('qc,kc->qk', output, output) / math.sqrt(output.shape[-1])
            outputs_reid = outputs_reid * max(self.reid_temp.weight.exp(), 1)
            
        else:
            use_norm = True or (task == 'grounding')
            output_norm = F.normalize(self.decoder_norm(output), p=2, dim=-1) if use_norm else self.decoder_norm(output)
            if self.prompt_as_queries:
                output_p = output_norm[self.num_queries:]
                outputs_reid = torch.einsum('qNC,kNC->qkN', output_norm, output_p)
                if not use_norm:
                    outputs_reid = outputs_reid / math.sqrt(output.shape[-1])
                outputs_reid = rearrange(outputs_reid, 'q k (N T) -> N T q k', T=t).mean(1)
                outputs_reid = outputs_reid * max(self.reid_temp.weight.exp(), 1)

                if task == 'grounding':
                    # for each target, retrieval the matched mask from learnable queries
                    l4p_indices = outputs_reid[:, :self.num_queries].flatten(0, -2).argmax(0)  # k
                    outputs_mask[:, self.num_queries:] = (outputs_mask[:, self.num_queries:] + outputs_mask[:, l4p_indices]) / 2.

            else:
                outputs_reid = [None] * bs
        
        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [BT, Q, H*W] -> [BT, h, Q, H*W] -> [B*T*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask.flatten(0, 1), 
            size=attn_mask_target_size, 
            mode="bilinear", 
            align_corners=False
        )
            
        attn_mask = rearrange(attn_mask, '(b q) t h w -> (b t) q (h w)', b=b)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, outputs_reid
    
    def prompt_image_attention_mask(self, attn_mask, attn_mask_target_size, num_frames, targets):
        if 'masks' not in targets[0] or targets[0]['masks'].nelement() == 0:
            return attn_mask
            
        # only for inference now
        gt_masks = torch.stack([
            targets_per_video['masks'][:, -num_frames:] for targets_per_video in targets
        ]) 
        # B, q_p, T, H, W -> B, T, q_p, H, W -> BT, q_p, HW 
        gt_masks = F.interpolate(
            gt_masks.transpose(1,2).flatten(0,1).float(), 
            size=attn_mask_target_size, 
            mode="bilinear", 
            align_corners=False
        ).flatten(-2)
        BT, num_gt_insts, L = gt_masks.shape

        # remain one pixel at least for each instance mask
        gt_masks = gt_masks.flatten(0, -2)
        max_logits, pixel_idxs = gt_masks.max(-1)
        batch_idx = torch.arange(gt_masks.shape[0], device=max_logits.device)
        batch_idx = batch_idx[max_logits > 0]
        pixel_idxs = pixel_idxs[max_logits > 0]
        gt_masks[batch_idx, pixel_idxs] = 1.
        # BT, q_p, HW -> BTh, q_p, HW
        gt_masks_not = (gt_masks.reshape(BT, num_gt_insts, L).repeat(1,self.num_heads,1, 1).flatten(0, 1) < 0).bool()
        
        attn_mask[:, self.num_queries:] = gt_masks_not & attn_mask[:, self.num_queries:]
        return attn_mask.detach()
    
    def forward_prompt_encoder(
        self, src, pos, size_list, targets, num_frames=None, prompt_type=None, use_all_prev_frames=False
    ):
        """
         src: [H_lW_lxNTxC], l=1,2,3,4, multi-scale features after pixel encoder
         pos: [H_lW_lxNTxC], l=1,2,3,4, positioanl embeddings of img emb
         size_list: [(H_0, W_0), ...], multi-scale feature sizes
         num_frames: number of frames 
         prompt_type: type of prompt, e.g., 'visual' or 'textual'
        """
        if num_frames is None:
            num_frames = self.num_frames
        device = src[0].device
        tasks_batch = [target_per_video['task'] for target_per_video in targets]
        assert sum([tasks_batch[0] == task for task in tasks_batch]) == len(tasks_batch)

        prompt_feats_dense, prompt_pe_dense = None, None
        l2v_attn_weights_list = None
        if tasks_batch[0] == 'sot' or targets[0]["prompt_type"] == 'visual' or prompt_type == 'visual':
            prompt_tuple = self.visual_prompt_sampler.process_per_batch(
                src, pos, size_list, targets, self.training, use_all_prev_frames=use_all_prev_frames
            )
            prompt_pe_dense, prompt_feats_dense = prompt_tuple[:2]
            if prompt_feats_dense is None:
                query_embed_prompt = None
                output_prompt = None
            else:
                # remove blank prompt, whose embeddings are zero-vector
                isnot_blank = torch.logical_not((prompt_feats_dense == 0).all(dim=-1)).unsqueeze(-1).sum(1).clamp(min=1)
                # num_inst x NT x C
                prompt_feats_mean = prompt_feats_dense.sum(1) / isnot_blank 
                prompt_pe_mean = prompt_pe_dense.sum(1) / isnot_blank 
                if (self.training and (torch.rand(1) > 0.5)): 
                    query_embed_prompt = prompt_pe_mean  # pos emb as query emb
                else:
                    query_embed_prompt = prompt_feats_mean  # use content feat as query emb
                output_prompt = prompt_feats_mean + self.prompt_sot.weight.view(1,1,-1)
                
                if not self.training and "prompt_feats" in targets[0]:
                    prompt_pe_dense, prompt_feats_dense = \
                        self.extract_prompt_features_from_memoey_pool(targets, num_frames)

        else:
            if tasks_batch[0] == 'detection':
                prompt_feats_batch = []
                for target_per_video in targets:
                    dataset_name = target_per_video["dataset_name"]
                    # Step 1: produce prompt features based on categories
                    assert dataset_name in combined_datasets_category_info
                    num_classes, start_idx = combined_datasets_category_info[dataset_name]
                    clip_cls_text_emb = self.clip_cls_text_emb[start_idx:start_idx + num_classes].to(device)  # num_classes x 640
                    assert len(clip_cls_text_emb) == num_classes, \
                        f'Dismatch numbers of class, {len(clip_cls_text_emb)} and {num_classes}'

                    # Note: labels start from 1 instead of 0
                    if self.training:
                        prompt_gt_labels = target_per_video["prompt_gt_labels"]
                        prompt_cls_text_emb = clip_cls_text_emb[prompt_gt_labels - 1]
                    else:
                        prompt_cls_text_emb = clip_cls_text_emb

                    # num_classes x 640 -> num_classes x C -> num_classes x T x C
                    prompt_cls_text2vis_emb = self.text2vis_projection(self.text_norm(prompt_cls_text_emb))
                    prompt_feats = prompt_cls_text2vis_emb[:, None].repeat(1, num_frames, 1)  
                    prompt_feats_batch.append(prompt_feats)

                prompt_feats_batch = torch.stack(prompt_feats_batch, dim=1).flatten(1, 2)  # num_classes x NT x C
                if self.text_prompt_to_image_enable:
                    # l2v_attn_weights: [N x num_classes x T x Hl x Wl], l=0,1,2
                    prompt_feats_batch, l2v_attn_weights_list = self.forward_lang_to_vision(
                        prompt_feats_batch, src, size_list, num_frames, tasks_batch[0]
                    )  
                
                prompt_feats_dense = prompt_feats_batch.unsqueeze(1)
                query_embed_prompt = prompt_feats_batch
                output_prompt = prompt_feats_batch + self.prompt_detection.weight.view(1,1,-1)

            elif tasks_batch[0] == 'grounding':
                prompt_feats_batch = []
                for target_per_video in targets:
                    # Step 1: produce prompt features based on expression
                    exp_word_len = target_per_video["exp_word_len"]
                    exp_word_feats = target_per_video["exp_word_feats"][..., :num_frames, :]          # num_exp x 77 x T x 640
                    exp_sentence_feats = target_per_video["exp_sentence_feats"][..., :num_frames, :]  # num_exp x T x 640
                    num_exps, len_sentence = exp_word_feats.shape[:2]

                    # Step 2: prompt-aware mask features
                    exp_feats = torch.cat([exp_sentence_feats[:, None], exp_word_feats], dim=1).flatten(0, 1)
                    # exp_word_feats: num_exp * (1+77) x T x 640 -> num_exp * (1+77) x T x C
                    prompt_feats_exp = self.text2vis_projection(self.text_norm(exp_feats))
                    prompt_feats_batch.append(prompt_feats_exp)

                prompt_feats_batch = torch.stack(prompt_feats_batch, dim=1).flatten(1, 2)  # num_exp * (1+77) x NT x C 
                if self.text_prompt_to_image_enable:
                    # l2v_attn_weights: [N x num_exps x T x Hl x Wl], l=0,1,2
                    prompt_feats_batch, l2v_attn_weights_list = self.forward_lang_to_vision(
                        prompt_feats_batch, src, size_list, num_frames, tasks_batch[0]
                    )
                # num_exp*L x NT x C -> num_exp x T*(1+77) x N x 1 x C -> num_exp x T*(1+77) x NT x C
                prompt_feats_batch = rearrange(
                    prompt_feats_batch, '(K L) (N T) C -> K T L N 1 C', K=num_exps, L=len_sentence+1, T=num_frames
                )[:,:,:32]  # not supoort long text yet
                prompt_feats_dense = prompt_feats_batch.flatten(1,2).repeat(1,1,1,num_frames,1).flatten(2,3) # num_exp x T*(1+77) x NT x C
                
                # only use the sentence token output from the CLIP text encoder
                prompt_feats_sentence = prompt_feats_batch.mean(1)[:,0].repeat(1,1,num_frames,1).flatten(1,2)  # num_exp x NT x C
                query_embed_prompt = prompt_feats_sentence 
                output_prompt = prompt_feats_sentence + self.prompt_grounding.weight.view(1,1,-1)

                if not self.training and 'masks' in targets[0] and self.use_visual_prompts_grounding:
                    prompt_feats_dense_vis, prompt_pe_dense_vis = self.visual_prompt_sampler.process_per_batch(
                        src, pos, size_list, targets, self.training, use_all_prev_frames=True
                    )[:2]
                    if prompt_feats_dense_vis is not None:
                        # num_exp x R x T x C -> num_exp x TR x C -> num_exp x TR x T x C
                        prompt_feats_dense_vis = prompt_feats_dense_vis.transpose(1,2).flatten(1,2).unsqueeze(2).repeat(1,1,num_frames,1)
                        prompt_feats_dense = torch.cat([prompt_feats_dense_vis, prompt_feats_dense], dim=1)
                    if prompt_pe_dense is not None and prompt_pe_dense_vis is not None:
                        prompt_pe_dense = torch.cat([prompt_pe_dense_vis, prompt_pe_dense], dim=1)
                
            else:
                raise ValueError
        
        # merge prompts from previous clips for stage3 training
        if self.training and (len(targets) == 1) and "prompt_feats" in targets[0] and prompt_feats_dense is not None:
            # print(targets[0]["prompt_feats"].shape, prompt_feats_dense.shape)
            prompt_feats_dense = torch.cat([targets[0]["prompt_feats"], prompt_feats_dense], dim=1)
            if targets[0]["prompt_pe"] is None or prompt_pe_dense is None:
                prompt_pe_dense = None
            else:
                prompt_pe_dense = torch.cat([targets[0]["prompt_pe"], prompt_pe_dense], dim=1)
        
        return output_prompt, query_embed_prompt, prompt_feats_dense, prompt_pe_dense, l2v_attn_weights_list
    
    def forward_lang_to_vision(self, prompt_feats, src, size_list, num_frames, task_type):
        """
        Args:
            prompt_feats: (1+77)*K x NT x C, K is the number of expressions for detection and grouding tasks, respectively.
            src: [H_lW_l x NT x C], l=0,1,2
            pos: [H_lW_l x NT x C], l=0,1,2
            size_list: [(H_0, W_0), (H_1, W_1), (H_2, W_2)]

        return: 
            l2v_prompt_feats: (1+77)*K x NT x C, text features after lang2imge interaction
            l2v_attn_weights: [N x K x T x h_l x W_l], l=0,1,2; averaged attnetion weights in all heads
        """
        assert task_type in {"grounding", "detection"}, "The prompt sould be category names / expressions!"
        
        src_flatten = torch.cat(src)  # LxNTxC
        l2v_prompt_feats, l2v_attn_weights_ori = self.lang2vision_cross_attention_layer(
            prompt_feats, src_flatten, 
        )

        # split l2v attention weights into multi-scale shapes for pixel-wise l2v-loss
        l2v_attn_weights = l2v_attn_weights_ori.clone()  # NT x (1+77)*K x L
        # scaling the max value to 1
        l2v_attn_weights = l2v_attn_weights / torch.max(l2v_attn_weights, dim=-1, keepdim=True)[0].clamp(min=1e-6)
        if task_type == 'grounding':
            # only use the sentence token
            l2v_attn_weights = rearrange(l2v_attn_weights, 'N (l k) L -> N l k L', l=78)[:, 0]
        l2v_attn_weights = torch.split(l2v_attn_weights, [src_i.shape[0] for src_i in src], dim=-1)
        l2v_attn_weights = [
            rearrange(weights, '(N T) k (h w) -> N k T h w', T=num_frames, h=h, w=w) 
            for weights, (h, w) in zip(l2v_attn_weights, size_list)
        ]
        return l2v_prompt_feats, l2v_attn_weights
    
    @torch.no_grad()
    def extract_prompt_features_from_memoey_pool(self, targets, num_frames):
        targets_per_video = targets[0]

        # extract prompt features from memory pool
        num_gt_insts, _, e_idx = targets_per_video["prompt_feats"].shape[:3]  # not include the last frame (blank)
        s_idx = max(0, e_idx - self.num_prev_frames_memory + 1)
        first_appear_frame_idxs = targets_per_video["first_appear_frame_idxs"].clone()
        first_appear_frame_idxs[first_appear_frame_idxs >= e_idx-1] = -1
        prompt_feats_first = targets_per_video["prompt_feats"][range(num_gt_insts),:, first_appear_frame_idxs]
        prompt_feats_prev = targets_per_video["prompt_feats"][:,:,s_idx:] 
        prompt_feats_prev = torch.cat([prompt_feats_first.unsqueeze(2), prompt_feats_prev], dim=2)
        prompt_pe_first = targets_per_video["prompt_pe"][range(num_gt_insts),:, first_appear_frame_idxs]
        prompt_pe_prev = targets_per_video["prompt_pe"][:,:,s_idx:] 
        prompt_pe_prev = torch.cat([prompt_pe_first.unsqueeze(2), prompt_pe_prev], dim=2)

        prompt_feats_dense = prompt_feats_prev.flatten(1,2).unsqueeze(2).repeat(1,1,num_frames,1)
        prompt_pe_dense = prompt_pe_prev.flatten(1,2).unsqueeze(2).repeat(1,1,num_frames,1)

        return prompt_pe_dense, prompt_feats_dense
    
    @torch.no_grad()
    def generate_self_attn_mask(self, bs, t, num_queries_lp, device, dataset_name, task):
        if self.maskdec_self_attn_mask_type in {'none', 'all'}:
            return None
        
        self_attn_mask = torch.ones((num_queries_lp*t, num_queries_lp*t), device=device)  # QT x QT
        self_attn_mask[:self.num_queries*t, :self.num_queries*t] = 0
        if self.maskdec_self_attn_mask_type == 'sep-blocked' or task == 'grounding':
            num_queries_p = num_queries_lp - self.num_queries 
            if num_queries_p > 0:
                prompt_attn_idx = torch.as_tensor([
                        [i, j] for i in range(t) for j in range(t
                    )])[None] + torch.arange(0, num_queries_p).reshape(-1, 1, 1) * t
                prompt_attn_idx = (prompt_attn_idx + self.num_queries*t).flatten(0, 1).to(device)
                prompt_attn_y, prompt_attn_x = torch.unbind(prompt_attn_idx, dim=-1)
                self_attn_mask[prompt_attn_y, prompt_attn_x] = 0
        elif self.maskdec_self_attn_mask_type == 'sep':
            self_attn_mask[self.num_queries*t:, self.num_queries*t:] = 0
        elif self.maskdec_self_attn_mask_type == 'sep-l2p':
            self_attn_mask[self.num_queries*t:] = 0
        else:
            raise ValueError

        self_attn_mask = self_attn_mask[None].repeat(self.num_heads*bs, 1, 1).bool().detach()  # h, QT, QT
        return self_attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_reid, output_embds):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_reid_logits": c, "pred_embds": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_reid[:-1], output_embds[:-1])
            ]
        else:
            return [
                {"pred_masks": b, "pred_reid_logits": c, "pred_embds": d} 
                for b, c, d in zip(outputs_seg_masks[:-1], outputs_reid[:-1], output_embds[:-1])
            ]

    def plot_mask_features(self, x, targets):
        import matplotlib.pyplot as plt
        image_size_with_pad = targets[0]['inter_image_size']
        image_size = targets[0]['image_size']
        rate_h = float(image_size[0]) / image_size_with_pad[0]
        rate_w = float(image_size[1]) / image_size_with_pad[1]
        x_h, x_w = x.shape[-2:]
        x_h_unpad = int(x_h * rate_h)
        x_w_unpad = int(x_w * rate_w)
        
        x = x[..., ::16, :x_h_unpad, :x_w_unpad]

        save_dir = 'output/visual/mask_features/'
        vid_id = int(torch.rand(1) * 1000000)
        save_path = os.path.join(save_dir, str(vid_id))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # x: T x C x H x W
        for c, x_ic in enumerate(x.transpose(0, 1)):
            # normalize the map for better visualization
            x_ic = (x_ic - x_ic.min()) / (x_ic.max() - x_ic.min())
            x_ic_np = x_ic.flatten(0,1).cpu().numpy()
            plt.imshow(x_ic_np)
            plt.title(str(c))
            plt.savefig(os.path.join(save_path, str(c)+'.jpg'), bbox_inches='tight')




        