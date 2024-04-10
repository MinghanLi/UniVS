CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/semantic_extraction/univs_swinb_internvid_flt.yaml \
    --eval-only \
    MODEL.UniVS.MASKDEC_SELF_ATTN_MASK_TYPE 'sep' \
    MODEL.UniVS.TEST.NUM_PREV_FRAMES_MEMORY 5 \
    INPUT.SAMPLING_FRAME_NUM 5 \
    INPUT.MIN_SIZE_TEST 512 \
    INPUT.LSJ_AUG.SQUARE_ENABLED False \
    DATASETS.TEST '("internvid-flt-1", )' \
    MODEL.BoxVIS.EMA_ENABLED True \
    MODEL.WEIGHTS output/stage3/univs_r50_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/semantic_extraction