# Convert custom videos to COCO annotations
python datasets/data_utils/custom_videos/convert_custom_videos_to_coco_test.py 

python train_net.py \
    --num-gpus 1 \
    --dist-url tcp://127.0.0.1:50191 \
    --config-file configs/univs_inf_custom_videos/univs_swinb_vps_c1+univs_entity.yaml \
    --eval-only \
    INPUT.MIN_SIZE_TEST 512 \
    MODEL.WEIGHTS pretrained/univs_v2_cvpr/univs_swinb_stage3_f7_wosquare_ema.pth \
    OUTPUT_DIR datasets/custom_videos/results/