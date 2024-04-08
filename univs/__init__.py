# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.

# config
from .config import add_univs_config

from .utils.copy_TeacherNet_weights import copy_TeacherNet_weights

from .data import *
from .modeling import *
from .evaluation import *
from .inference import *

from .prepare_targets import PrepareTargets
from .univs_prompt import UniVS_Prompt
from .univs_prompt_longvideo import UniVS_Prompt_LongVideo