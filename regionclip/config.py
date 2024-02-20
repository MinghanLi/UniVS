from detectron2.config import CfgNode as CN


def add_regionclip_config(cfg):
    cfg.MODEL.CLIP = CN()

    cfg.MODEL.CLIP.BACKBONE_FREEZE_AT = 2
    cfg.MODEL.CLIP.RESNETS_OUT_FEATURES = ["res4"]
    cfg.MODEL.CLIP.RESNETS_DEPTH = 200
    cfg.MODEL.CLIP.RESNETS_RES2_OUT_CHANNELS = 256

    cfg.MODEL.CLIP.WEIGHTS = "pretrained/regionclip/regionclip/regionclip_pretrained-cc_rn50x4_only_lang_encoder.pth"  # option: all configs of pretrained RegionCLIP
    cfg.MODEL.CLIP.CONFIG_FILE = "configs/regionclip/CLIP_fast_rcnn_R50_C4_sam.yaml"

    cfg.MODEL.CLIP.CROP_REGION_TYPE = "GT"  # options: "GT", "RPN"
    cfg.MODEL.CLIP.BB_RPN_WEIGHTS = None  # the weights of pretrained MaskRCNN
    cfg.MODEL.CLIP.IMS_PER_BATCH_TEST = 8  # the #images during inference per batch

    cfg.MODEL.CLIP.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    cfg.MODEL.CLIP.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

    cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER = True  # if True, use the CLIP text embedding as the classifier's weights
    cfg.MODEL.CLIP.TEXT_EMB_PATH = "pretrained/regionclip/concept_emb/googlecc_nouns_6250_emb_rn50x4.pth"

    cfg.MODEL.CLIP.BG_CLS_LOSS_WEIGHT = None  # if not None, it is the loss weight for bg regions
    cfg.MODEL.CLIP.ONLY_SAMPLE_FG_PROPOSALS = False  # if True, during training, ignore all bg proposals and only sample fg proposals

    cfg.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES = None  # if an integer, it is #all_cls in test
    cfg.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH = None  # if not None, enables the openset/zero-shot training, the category embeddings during test

    cfg.MODEL.CLIP.CLSS_TEMP = 0.01  # normalization + dot product + temperature
    cfg.MODEL.CLIP.FOCAL_SCALED_LOSS = None  # if not None (float value for gamma), apply focal loss scaling idea to standard cross-entropy loss

    cfg.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL = True  # if True, pretrain model using image-text level matching
    cfg.MODEL.CLIP.PRETRAIN_ONLY_EOT = False  # if True, use end-of-token emb to match region features, in image-text level matching
    cfg.MODEL.CLIP.PRETRAIN_RPN_REGIONS = None  # if not None, the number of RPN regions per image during pretraining
    cfg.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS = None  # if not None, the number of regions per image during pretraining after sampling, to avoid overfitting
    cfg.MODEL.CLIP.GATHER_GPUS = False  # if True, gather tensors across GPUS to increase batch size
    cfg.MODEL.CLIP.GRID_REGIONS = False  # if True, use grid boxes to extract grid features, instead of object proposals
    cfg.MODEL.CLIP.CONCEPT_POOL_EMB = None  # if not None, it provides the file path of embs of concept pool and thus enables region-concept matching
    cfg.MODEL.CLIP.CONCEPT_THRES = None  # if not None, the threshold to filter out the regions with low matching score with concept embs, dependent on temp (default: 0.01)

    cfg.MODEL.CLIP.TEXT_EMB_DIM = 640  # the dimension of precomputed class embeddings
    cfg.INPUT_DIR = "./pretrained/regionclip/concept_emb/"  # the folder that includes the images for region feature extraction
    cfg.MODEL.CLIP.GET_CONCEPT_EMB = False  # if True (extract concept embedding), a language encoder will be created
    cfg.CONCEPTS_FILE = 'googlecc_nouns_6250.txt'

    # ---------------------------------------------------------------------------- #
    # ROI HEADS options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.ROI_HEADS = CN()
    cfg.MODEL.ROI_HEADS.NAME = "CLIPRes5ROIHeads"
    # Number of foreground classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6250
    # Names of the input feature maps to be used by ROI heads
    # Currently all heads (box, mask, ...) use the same input feature map list
    # e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
    # IOU overlap ratios [IOU_THRESHOLD]
    # Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
    # Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    cfg.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
    # RoI minibatch size *per image* (number of regions of interest [ROIs])
    # Total number of RoIs per training minibatch =
    #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
    # E.g., a common configuration is: 512 * 16 = 8192
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

    # Only used on test mode

    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 18
    # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
    # balance obtaining high recall with not having too many low precision
    # detections that will slow down inference post processing steps (like NMS)
    # A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
    # inference.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    # Overlap threshold used for non-maximum suppression (suppress boxes with
    # IoU >= this threshold)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    # If True, augment proposals with ground-truth boxes before sampling proposals to
    # train ROI heads.
    cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    # See soft NMS paper for definition of these options
    cfg.MODEL.ROI_HEADS.SOFT_NMS_METHOD = "gaussian"  # "linear"
    cfg.MODEL.ROI_HEADS.SOFT_NMS_SIGMA = 0.5
    # For the linear_threshold we use NMS_THRESH_TEST
    cfg.MODEL.ROI_HEADS.SOFT_NMS_PRUNE = 0.001

    # VIS tasks
    cfg.INPUT.SQUARE_SIZE = False
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.SEGMENT_ANYTHING = False