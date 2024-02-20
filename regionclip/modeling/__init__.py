from .backbone.clip_backbone import build_clip_resnet_backbone

from .meta_arch.clip_rcnn import CLIPFastRCNN, build_CLIPFastRCNN

from .roi_heads.clip_roi_heads import CLIPRes5ROIHeads
from .roi_heads.fast_rcnn import FastRCNNOutputLayers