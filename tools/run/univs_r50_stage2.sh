CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --num-gpus 1 \
  --dist-url tcp://127.0.0.1:50159 \
  --config-file configs/univs/univs_r50_stage2.yaml \
  --resume \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.BASE_LR 0.00005 \
  INPUT.SAMPLING_FRAME_NUM 2 \
  INPUT.SAMPLING_FRAME_WINDOE_NUM 2 \
  INPUT.SAMPLING_FRAME_VIDEO_NUM 2 \
  INPUT.LSJ_AUG.SQUARE_ENABLED False \
  TEST.EVAL_PERIOD 5000 \
  MODEL.WEIGHTS pretrained/univs/r50/v1.1/univs_prompt_R50_bs8_f3_video_4x_200queries_c1+univs+casa_prompt_sa_frozenbb_v2.pth \
  OUTPUT_DIR output/v2/test/ #univs_r50_stage2