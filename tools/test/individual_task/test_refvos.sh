CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/refvos/univs_swinl_refvos_davis_c1+univs.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep-blocked' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 10 \
    INPUT.SAMPLING_FRAME_NUM 10 \
    INPUT.MIN_SIZE_TEST 640 \
    DATASETS.TEST '("rvos-refdavis-val-0", "rvos-refdavis-val-1", "rvos-refdavis-val-2", "rvos-refdavis-val-3")' \
    MODEL.WEIGHTS output/stage2/univs_swinl_stage2.pth \
    OUTPUT_DIR output/stage2/univs_swinl_stage2/inf_training_final/refvos/davis_dev/