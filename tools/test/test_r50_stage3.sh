CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vis/univs_R50_yt21_c1+univs_entity.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 5 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    INPUT.LSJ_AUG.SQUARE_ENABLED False \
    DATASETS.TEST '("ytvis_2021_dev", )' \
    MODEL.WEIGHTS output/stage3/univs_r50_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/stage3/univs_r50_stage3_wosquare/inf_training_final/vis/yt21_dev/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vis/univs_R50_ovis_c1+univs_entity.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 5 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    INPUT.LSJ_AUG.SQUARE_ENABLED False \
    DATASETS.TEST '("ovis_dev", )' \
    MODEL.WEIGHTS output/stage3/univs_r50_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/stage3/univs_r50_stage3_wosquare/inf_training_final/vis/ovis_dev/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vss/univs_R50_vss_c1+univs_entity.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    INPUT.LSJ_AUG.SQUARE_ENABLED False \
    DATASETS.TEST '("vspw_vss_video_dev", )' \
    MODEL.WEIGHTS output/stage3/univs_r50_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/stage3/univs_r50_stage3_wosquare/inf_training_final/vss/vspw/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vps/univs_R50_vps_c1+univs_entity.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 5 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    INPUT.LSJ_AUG.SQUARE_ENABLED False \
    DATASETS.TEST '("vipseg_panoptic_dev", )' \
    MODEL.WEIGHTS output/stage3/univs_r50_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/stage3/univs_r50_stage3_wosquare/inf_training_final/vps/vipseg_dev/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vos/univs_R50_vos_davis17_c1+univs.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 5 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    INPUT.LSJ_AUG.SQUARE_ENABLED False \
    MODEL.WEIGHTS output/stage3/univs_r50_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/stage3/univs_r50_stage3_wosquare/inf_training_final/vos/davis17/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/pvos/univs_R50_pvos_c1+univs.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 5 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    INPUT.LSJ_AUG.SQUARE_ENABLED False \
    DATASETS.TEST '("pvos_viposeg_dev", )' \
    MODEL.WEIGHTS output/stage3/univs_r50_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/stage3/univs_r50_stage3_wosquare/inf_training_final/pvos/viposeg_dev/ \

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/refvos/univs_R50_refvos_davis_c1+univs.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep-blocked' \
    MODEL.UniVS.TEST.ENABLED_PREV_VISUAL_PROMPTS_FOR_GROUNDING False \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 5 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    INPUT.LSJ_AUG.SQUARE_ENABLED False \
    DATASETS.TEST '("rvos-refdavis-val-1", )' \
    MODEL.WEIGHTS output/stage3/univs_r50_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/stage3/univs_r50_stage3_wosquare/inf_training_final/refvos/davis_dev/ 