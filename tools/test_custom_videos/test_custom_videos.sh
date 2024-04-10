CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf_custom_videos/univs_swinb_vps_c1+univs_entity.yaml \
    --eval-only \
    INPUT.MIN_SIZE_TEST 512 \
    MODEL.WEIGHTS output/stage3/univs_swinb_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR output/inf/custom_videos