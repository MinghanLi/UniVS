CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --num-gpus 4 \
  --dist-url tcp://127.0.0.1:50159 \
  --config-file configs/univs/univs_swinb_stage3.yaml \
  --resume \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.000025 \
  INPUT.SAMPLING_FRAME_NUM 5 \
  INPUT.SAMPLING_FRAME_WINDOE_NUM 5 \
  INPUT.SAMPLING_FRAME_VIDEO_NUM 7 \
  INPUT.LSJ_AUG.SQUARE_ENABLED False \
  TEST.EVAL_PERIOD 5000 \
  MODEL.WEIGHTS output/v2/univs_swinb_stage2/model_final.pth \
  OUTPUT_DIR output/v2/univs_swinb_stage3_f7