import os
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt


def calculate_mask_quality_scores(mask_pred, threshold=1):
    # mask_pred is the logits, before activation
    scores_mask = (mask_pred > threshold).flatten(1).sum(-1) / (mask_pred > -threshold).flatten(1).sum(-1).clamp(min=1)
    return scores_mask


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConvertSemanticFeatureToMask(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        mask_dim=256,
        text_emb_dim=640,
        apply_cls_thres=0.65,
        apply_mask_quality_thres=0.85,
        temporal_stride=10,
        clip_class_embed_path='datasets/concept_emb/combined_datasets_cls_emb_rn50x4.pth',
        pretrained_ckpt='pretrained/univs_v2_cvpr/univs_swinb_stage3_f7_wosquare_ema.pth',
        device='cuda',
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.decoder_norm = nn.LayerNorm(hidden_dim).to(self.device)
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3).to(self.device)
        # vis2lang head and lang2vis head
        self.vis2text_projection = nn.Linear(hidden_dim, text_emb_dim).to(self.device)
        # text embeddings of category names output from CLIP Text Encoder
        self.clip_cls_text_emb = torch.load(clip_class_embed_path, map_location='cpu').to(self.device)
        self.cls_temp = nn.Embedding(1, 1).to(self.device)

        self.apply_cls_thres = apply_cls_thres
        self.apply_mask_quality_thres = apply_mask_quality_thres
        self.temporal_stride = temporal_stride

        self.load_pretrained_checkpoint(pretrained_ckpt)
    
    def load_pretrained_checkpoint(self, pretrained_ckpt):
        pretrained = torch.load(pretrained_ckpt, map_location=self.device)
        if "model" in pretrained:
            pretrained_weights = pretrained["model"]
        elif isinstance(pretrained, dict):
            pretrained_weights = pretrained
        else:
            raise ValueError
        pretrained_keys = list(pretrained_weights.keys())
        
        # Prepare the current model's state dictionary for updates
        current_state = self.state_dict()

        # Update current model's state dictionary with weights from the pretrained model
        for name, param in current_state.items():
            matched_name = None
            for k in pretrained_keys:
                if name == k.replace("sem_seg_head.predictor.", "") and param.size() == pretrained_weights[k].size():
                    matched_name = k
                    break
            if matched_name is not None:
                print(f"Matching {name} ==> {matched_name} from pretrained weights.")
                current_state[name].copy_(pretrained_weights[matched_name])
            else:
                print(f"Skipping {name} as it is not in the pretrained model or size mismatch.")

        # Load the updated state dictionary back to the model
        self.load_state_dict(current_state)        

    def convert(self, mask_feats, obj_tokens, only_high_conf_masks=True):
        # mask_feats: [T, C, H/32, W/32]; obj_tokens: [T, C, num_obj_tokens]
        obj_tokens = self.decoder_norm(obj_tokens.transpose(1, 2))  

        cls_logits = self.vis2text_projection(obj_tokens)
        CLIP_class = F.normalize(self.clip_cls_text_emb, p=2, dim=-1)
        cls_logits = F.normalize(cls_logits, p=2, dim=-1)
        cls_logits = torch.einsum('tnc,kc->tnk', cls_logits, CLIP_class)
        cls_logits = cls_logits * self.cls_temp.weight.exp()
        cls_logits = cls_logits.transpose(0, 1)

        mask_embed = self.mask_embed(obj_tokens)  
        mask_logits = torch.einsum("tnc,tchw->tnhw", mask_embed, mask_feats)
        mask_logits = mask_logits.transpose(0, 1)

        if only_high_conf_masks:
            cls_scores = cls_logits.sigmoid()
            is_high_conf = cls_scores[..., 1000:].flatten(1).max(1)[0] > self.apply_cls_thres
            # memory efficient
            mask_quality_scores = calculate_mask_quality_scores(mask_logits[:, ::self.temporal_stride]) 
            is_high_quality = mask_quality_scores > self.apply_mask_quality_thres
            is_high_indices = torch.nonzero(is_high_conf & is_high_quality).reshape(-1)
            # print(cls_scores[..., 1000:].flatten(1).max(1)[0][is_high_indices])
            # print(mask_quality_scores[is_high_indices])
            return cls_logits[is_high_indices], mask_logits[is_high_indices], is_high_indices

        return cls_logits, mask_logits, torch.arange(mask_logits.shape[0])


def plot_masks(mask_logits):
    save_dir = 'output/visual/semantic_masks/'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(0, mask_logits.shape[0], 10):
        m_numpy = rearrange(mask_logits[i:i+10, ::20].gt(0.), 'n t h w -> (n h) (t w)').detach().cpu().numpy()
        plt.figure(figsize=(20, 12))
        plt.imshow(m_numpy, cmap='viridis')  # 'viridis' is a commonly used color map, but you can choose another
        # plt.colorbar()  # Adds a color bar to the side showing the scale
        plt.title('Visualized Feature Map')

        # Save the figure
        output_path = os.path.join(save_dir, str(i)+'.jpg')
        plt.savefig(output_path, format='jpg', dpi=300)  # dpi is optional, increase for higher resolution
        plt.clf()
        # Inform the user
        print(f"Figure saved as {output_path}")


if __name__  == "__main__":
    converter = ConvertSemanticFeatureToMask()

    mask_feat_suffix = "_compression_mask_features_32_1.pt"
    obj_token_suffix = "_obj_tokens_32_1.pt"

    video_name = "--hNhsGTd8s_00:02:04.680_00:02:14.680"
    video_dir = "datasets/internvid/semantic_extraction/InternVId-FLT_1"

    mask_feats = torch.load(os.path.join(video_dir, video_name + mask_feat_suffix))
    obj_tokens = torch.load(os.path.join(video_dir, video_name + obj_token_suffix))

    cls_logits, mask_logits, indices = converter.convert(mask_feats, obj_tokens)
    plot_masks(mask_logits)