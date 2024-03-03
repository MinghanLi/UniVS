#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A script for region feature extraction
"""

import os
import sys
from pathlib import Path
import torch

file = Path(__file__).resolve()
parent = str(file.parent)
parent_new = '/'.join(parent.split('/')[:parent.split('/').index('UniVS')+1])

sys.path.append(parent_new)
file = Path(__file__).resolve()
parent = str(file.parent)


from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from univs.modeling.language import build_clip_language_encoder, pre_tokenize, clean_strings
from config import add_clip_text_config
from convert_lang_encoder_weights import convert_lang_encoder_weights


def build_lang_encoder(cfg):
    """
    Given a config file, create a detector
    (refer to tools/train_net.py)
    """
    # create model
    lang_encoder = build_clip_language_encoder(cfg)
    print('Model arch:', lang_encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang_encoder = lang_encoder.to(device)

    # converted_weights = convert_lang_encoder_weights(cfg.MODEL.CLIP.WEIGHTS)
    # DetectionCheckpointer(lang_encoder, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     converted_weights, resume=False
    # )

    for p in lang_encoder.parameters(): p.requires_grad = False
    lang_encoder.eval()

    return lang_encoder


def extract_concept_embeddings(cfg, lang_encoder):
    # input concepts
    concept_file = os.path.join(cfg.INPUT_DIR, cfg.CONCEPTS_FILE)

    concept_feats = []
    with open(concept_file, 'r') as f:
        for line in f:
            concept = line.strip()
            with torch.no_grad():
                concept = [clean_strings(concept)]
                print('processing the class embedding of ', concept[0])
                # num_category x num_templates x length_of_text, i.e. 1 x 81 x 77
                token_embeddings = pre_tokenize([concept]).to(lang_encoder.device)[0]
                # input: 81 x 77, output: 81 x d_text
                text_features = lang_encoder.encode_text(token_embeddings)
                # average over all templates
                text_features = text_features.mean(0, keepdim=True)
                concept_feats.append(text_features)

    concept_feats = torch.stack(concept_feats, 0)  # N x d_text
    concept_feats = torch.squeeze(concept_feats)  # 1 x N x d_text
    saved_path = os.path.join(cfg.OUTPUT_DIR, cfg.CONCEPTS_FILE.replace('.txt', '_cls_emb_rn50x4.pth'))
    torch.save(concept_feats, saved_path)

    print("Save concept embeddings in:", saved_path)
    print(concept_feats.shape)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_clip_text_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    regionclip_cfg = setup(args)
    # create model
    lang_encoder = build_lang_encoder(regionclip_cfg)

    extract_concept_embeddings(regionclip_cfg, lang_encoder)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )