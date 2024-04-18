CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf_custom_videos/univs_swinb_text_c1+univs_entity.yaml \
    --eval-only \
    INPUT.MIN_SIZE_TEST 512 \
    MODEL.WEIGHTS pretrained/univs_v2_cvpr/univs_swinb_stage3_f7_wosquare_ema.pth \
    MODEL.UniVS.TEST.CUSTOM_VIDEOS_TEXT "[['a man is playing ice hockey', 'the goalie stick is held by a man', 'a flag on the left', 'the hockey goal cage']]" \
    OUTPUT_DIR datasets/custom_videos/results_text/