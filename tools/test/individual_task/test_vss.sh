CUDA_VISIBLE_DEVICES=1 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vss/univs_swinl_vss_c1+univs_entity.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    DATASETS.TEST '("vspw_vss_video_dev", )' \
    MODEL.WEIGHTS output/stage2/univs_swinl_stage2.pth \
    OUTPUT_DIR output/stage2/univs_swinl_stage2/inf_training_final/vss/vspw/