CUDA_VISIBLE_DEVICES=0,1 python train_net.py \
    --num-gpus 2 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf/vids/vis/univs_swinl_ovis_c1+univs_entity.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 6 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 640 \
   MODEL.WEIGHTS output/stage2/univs_swinl_stage2.pth \
    OUTPUT_DIR output/stage2/univs_swinl_stage2/inf_training_final/vis/ovis_dev/\
