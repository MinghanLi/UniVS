from detectron2.config import CfgNode as CN


def add_clip_text_config(cfg):
    cfg.MODEL.CLIP = CN()

    cfg.MODEL.CLIP.BACKBONE_FREEZE_AT = 2
    cfg.MODEL.CLIP.RESNETS_OUT_FEATURES = ["res5"]
    cfg.MODEL.CLIP.RESNETS_DEPTH = 200
    cfg.MODEL.CLIP.RESNETS_RES2_OUT_CHANNELS = 256

    # option: all configs of pretrained RegionCLIP
    cfg.MODEL.CLIP.WEIGHTS = "pretrained/regionclip/regionclip/regionclip_pretrained-cc_rn50x4_only_lang_encoder.pth"
    cfg.INPUT_DIR = "./datasets/concept_emb/"
    cfg.OUTPUT_DIR = './pretrained/regionclip/concept_emb/'
    cfg.CONCEPTS_FILE = 'coco_133.txt'