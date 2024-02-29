CUDA_VISIBLE_DEVICES=1 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/pvos/univs_swinl_pvos_c1+univs.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
    DATASETS.TEST '("pvos_viposeg_dev", )' \
    MODEL.WEIGHTS output/stage2/univs_swinl_stage2.pth \
    OUTPUT_DIR output/stage2/univs_swinl_stage2/inf_training_final/pvos/viposeg_dev/