from detectron2.config import CfgNode as CN


def add_univs_config(cfg):
    cfg.DATASETS.DATASET_RATIO = []
    cfg.DATASETS.DATALOADER_TYPE = 'iter'

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_WINDOE_NUM = -1 
    cfg.INPUT.SAMPLING_FRAME_VIDEO_NUM = -1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_RANGE_MOT = 20
    cfg.INPUT.SAMPLING_FRAME_RANGE_SOT = 20
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = []  # "brightness", "contrast", "saturation", "rotation"

    cfg.INPUT.MIN_SIZE_TRAIN = (512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.CROP = CN()
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute_range"
    cfg.INPUT.CROP.SIZE = (600, 1024)

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation'] # ['rotation', 'brightness']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = True
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (480, 1024)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = True
    cfg.INPUT.LSJ_AUG.SQUARE_ENABLED = True
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.25
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 4.0

    # VIT transformer backbone
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.PRETRAIN_IMG_SIZE = 1024
    cfg.MODEL.VIT.PATCH_SIZE = 16
    cfg.MODEL.VIT.EMBED_DIM = 768
    cfg.MODEL.VIT.DEPTH = 12
    cfg.MODEL.VIT.NUM_HEADS = 12
    cfg.MODEL.VIT.MLP_RATIO = 4.0
    cfg.MODEL.VIT.OUT_CHANNELS = 256
    cfg.MODEL.VIT.QKV_BIAS = True
    cfg.MODEL.VIT.USE_ABS_POS = True
    cfg.MODEL.VIT.USE_REL_POS = False
    cfg.MODEL.VIT.REL_POS_ZERO_INIT = False
    cfg.MODEL.VIT.WINDOW_SIZE = 0
    cfg.MODEL.VIT.GLOBAL_ATTN_INDEXES = ()
    cfg.MODEL.VIT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.VIT.USE_CHECKPOINT = False
    
    # language dim (640 in CLIP with R50x4 backbone)
    cfg.MODEL.SEM_SEG_HEAD.LANG_DIM = 640
    cfg.MODEL.SEM_SEG_HEAD.FROZEN_PIXEL_DECODER = False  # input_proj &transformer.encoder
    cfg.MODEL.SEM_SEG_HEAD.FROZEN_MASK_CONVS = False     # one 3*3 conv and two 1*1 convs
    cfg.MODEL.SEM_SEG_HEAD.FROZEN_PREDICTOR = False

    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskFormerHead"
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 133
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = [ "res2", "res3", "res4", "res5" ]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [ "res3", "res4", "res5" ]
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6

    # Mask2Former
    cfg.MODEL.MASK_FORMER.REID_WEIGHT = 0.25
    cfg.MODEL.MASK_FORMER.TEST.STABILITY_SCORE_THRESH = 0.0 # from SAM
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD_ENTITY = 0.5

    # BoxVIS
    cfg.MODEL.BoxVIS = CN()
    cfg.MODEL.BoxVIS.BoxVIS_ENABLED = False
    cfg.MODEL.BoxVIS.EMA_ENABLED = False
    cfg.MODEL.BoxVIS.PSEUDO_MASK_SCORE_THRESH = 0.5

    # Inference
    cfg.MODEL.BoxVIS.TEST = CN()
    cfg.MODEL.BoxVIS.TEST.LSJ_AUG_ENABLED = True
    cfg.MODEL.BoxVIS.TEST.ZERO_SHOT_INFERENCE = False
    cfg.MODEL.BoxVIS.TEST.TRACKER_TYPE = 'minvis'  # 'minvis' => frame-level tracker, 'mdqe' => clip-level tracker
    cfg.MODEL.BoxVIS.TEST.WINDOW_INFERENCE = False
    cfg.MODEL.BoxVIS.TEST.MULTI_CLS_ON = True
    cfg.MODEL.BoxVIS.TEST.APPLY_CLS_THRES = 0.05
    cfg.MODEL.BoxVIS.TEST.MERGE_ON_CPU = False

    # clip-by-clip tracking with overlapped frames
    cfg.MODEL.BoxVIS.TEST.NUM_FRAMES = 3
    cfg.MODEL.BoxVIS.TEST.NUM_FRAMES_WINDOW = 5
    cfg.MODEL.BoxVIS.TEST.NUM_MAX_INST = 50
    cfg.MODEL.BoxVIS.TEST.CLIP_STRIDE = 1

    # UniVS
    cfg.MODEL.UniVS = CN()
    cfg.MODEL.UniVS.PROMPT_TYPE = "category"
    cfg.MODEL.UniVS.CLIP_CLASS_EMBED_PATH = 'datasets/concept_emb/combined_datasets_cls_emb_rn50x4.pth'
    cfg.MODEL.UniVS.NUM_POS_QUERIES = 30   # maximum obejcts for each frame in visual prompts
    cfg.MODEL.UniVS.USE_CONTRASTIVE_LOSS = True
    
    # arch. parameters
    cfg.MODEL.UniVS.VISUAL_PROMPT_ENCODER = True
    cfg.MODEL.UniVS.TEXT_PROMPT_ENCODER = True
    cfg.MODEL.UniVS.LANGUAGE_ENCODER_ENABLE = True
    cfg.MODEL.UniVS.PROMPT_AS_QUERIES = True
    cfg.MODEL.UniVS.VISUAL_PROMPT_TO_IMAGE_ENABLE = True
    cfg.MODEL.UniVS.TEXT_PROMPT_TO_IMAGE_ENABLE = True
    cfg.MODEL.UniVS.MASKDEC_ATTN_ORDER = 'casa'  # 'casa' or 'saca'
    cfg.MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE = 'sep'  # 'all', 'sep', 'p2l-alpha', 'p2l-beta'
    cfg.MODEL.UniVS.DISABLE_LEARNABLE_QUERIES_SA1B = False
    cfg.MODEL.UniVS.VISUAL_PROMPT_PIXELS_PER_IMAGE = 32
    cfg.MODEL.UniVS.PROMPT_SELF_ATTN_LAYERS = -1
    cfg.MODEL.UniVS.POSITION_EMBEDDING_SINE3D = 'ArbitraryT' # "FixedT" or "ArbitraryT"

    cfg.MODEL.UniVS.TEST = CN()
    cfg.MODEL.UniVS.TEST.VIDEO_UNIFIED_INFERENCE_ENABLE = False
    cfg.MODEL.UniVS.TEST.VIDEO_UNIFIED_INFERENCE_QUERIES = 'prompt'
    cfg.MODEL.UniVS.TEST.VIDEO_UNIFIED_INFERENCE_ENTITIES = ''
    cfg.MODEL.UniVS.TEST.DISABLE_SEMANTIC_QUERIES = False
    cfg.MODEL.UniVS.TEST.BOX_NMS_THRESH = 0.75                    # from SAM
    cfg.MODEL.UniVS.TEST.TEMPORAL_CONSISTENCY_THRESHOLD = 0.05
    cfg.MODEL.UniVS.TEST.CLIP_STRIDE = 1
    cfg.MODEL.UniVS.TEST.DETECT_NEWLY_OBJECT_THRESHOLD = 0.05
    cfg.MODEL.UniVS.TEST.DETECT_NEWLY_INTERVAL_FRAMES = 1
    cfg.MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY = 5 
    cfg.MODEL.UniVS.TEST.ENABLED_PREV_FRAMES_MEMORY = True # False for stage2 but Ture for stage3
    cfg.MODEL.UniVS.TEST.ENABLED_PREV_VISUAL_PROMPTS_FOR_GROUNDING = False