# CUDA_VISIBLE_DEVICES=0 python train_net.py \
#     --num-gpus 1 \
#     --dist-url tcp://127.0.0.1:50191 \
#     --config-file configs/univs_inf/vids/vis/univs_swinb_yt21_c1+univs_entity.yaml \
#     --eval-only \
#     MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
#     MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
#     INPUT.SAMPLING_FRAME_NUM 5 \
#     INPUT.MIN_SIZE_TEST 640 \
#     DATASETS.TEST '("ytvis_2021_dev", )' \
#     MODEL.WEIGHTS output/v2/univs_swinb_stage3_f7/model_0099999.pth \
#     OUTPUT_DIR output/v2/univs_swinb_stage3_f7/inf_training_100k/vis/yt21_dev/ \

# CUDA_VISIBLE_DEVICES=0 python train_net.py \
#     --num-gpus 1 \
#     --dist-url tcp://127.0.0.1:50191 \
#     --config-file configs/univs_inf/vids/vis/univs_swinb_ovis_c1+univs_entity.yaml \
#     --eval-only \
#     MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
#     MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
#     INPUT.SAMPLING_FRAME_NUM 5 \
#     INPUT.MIN_SIZE_TEST 640 \
#     DATASETS.TEST '("ovis_dev", )' \
#     MODEL.WEIGHTS output/v2/univs_swinb_stage3_f7/model_0099999.pth \
#     OUTPUT_DIR output/v2/univs_swinb_stage3_f7/inf_training_100k/vis/ovis_dev/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vss/univs_swinb_vss_c1+univs_entity.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    DATASETS.TEST '("vspw_vss_video_dev", )' \
    MODEL.WEIGHTS output/v2/univs_swinb_stage3_f7/model_0099999.pth \
    OUTPUT_DIR output/v2/univs_swinb_stage3_f7/inf_training_100k/vss/vspw/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vps/univs_swinb_vps_c1+univs_entity.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    DATASETS.TEST '("vipseg_panoptic_dev", )' \
    MODEL.WEIGHTS output/v2/univs_swinb_stage3_f7/model_0099999.pth \
    OUTPUT_DIR output/v2/univs_swinb_stage3_f7/inf_training_100k/vps/vipseg_dev/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vos/univs_swinb_vos_davis17_c1+univs.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    MODEL.WEIGHTS output/v2/univs_swinb_stage3_f7/model_0099999.pth \
    OUTPUT_DIR output/v2/univs_swinb_stage3_f7/inf_training_100k/vos/davis17/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/pvos/univs_swinb_pvos_c1+univs.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    DATASETS.TEST '("pvos_viposeg_dev", )' \
    MODEL.WEIGHTS output/v2/univs_swinb_stage3_f7/model_0099999.pth \
    OUTPUT_DIR output/v2/univs_swinb_stage3_f7/inf_training_100k/pvos/viposeg_dev/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/refvos/univs_swinb_refvos_davis_c1+univs.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep-blocked' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 5 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    DATASETS.TEST '("rvos-refdavis-val-1", )' \
    MODEL.WEIGHTS output/v2/univs_swinb_stage3_f7/model_0099999.pth \
    OUTPUT_DIR output/v2/univs_swinb_stage3_f7/inf_training_100k/refvos/davis_dev/ 