from collections import OrderedDict

import torch
from torch import nn
import pickle


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPLangEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 out_features,
                 freeze_at,
                 ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp[
            0].weight.dtype  # torch.float32, not sure whether need to be fp16 in pretraining

    @property
    def device(self):
        return self.token_embedding.weight.device

    def encode_text(self, text, only_eot=True):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_eot = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if only_eot:
            return x_eot
        else:
            # return embeddings for all tokens, instead of the eot embedding as CLIP implementation below
            x_word = x @ self.text_projection
            return x_word, x_eot


def build_clip_language_encoder(cfg):
    """
    Create the CLIP language encoder instance from config.

    Returns:
        CLIP: a :class:`CLIP` instance.
    """
    # port standard ResNet config to CLIP ModifiedResNet
    freeze_at = cfg.MODEL.CLIP.BACKBONE_FREEZE_AT
    out_features = ['res5']  # includes the whole ResNet # cfg.MODEL.CLIP.OUT_FEATURES
    depth = cfg.MODEL.CLIP.RESNETS_DEPTH

    # default configs of CLIP
    embed_dim = {
        50: 1024,
        101: 512,
        200: 640,  # flag for ResNet50x4
    }[depth]
    context_length = 77
    vocab_size = 49408
    transformer_width = {
        50: 512,
        101: 512,
        200: 640,  # flag for ResNet50x4
    }[depth]
    transformer_heads = {
        50: 8,
        101: 8,
        200: 10,  # flag for ResNet50x4
    }[depth]
    transformer_layers = 12

    model = CLIPLangEncoder(
        embed_dim,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        out_features, freeze_at
    )

    load_checkpoint(model, cfg.MODEL.CLIP.WEIGHTS)

    return model

def load_checkpoint(model, model_weigths_path):
    if model_weigths_path is None or len(model_weigths_path) == 0:
        raise ValueError

    if model_weigths_path[-3:] == 'pkl':
        checkpoint = pickle.load(open(model_weigths_path, 'rb'))
    else:
        checkpoint = torch.load(model_weigths_path, map_location=torch.device('cpu'))
    print("Important: Load pretrained weights for Language Model!!!")
    model.load_state_dict(checkpoint['model'], strict=True)
