CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
  --num-gpus 8 \
  --dist-url tcp://127.0.0.1:50159 \
  --config-file configs/univs/univs_r50_stage2.yaml \
  --resume \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.BASE_LR 0.00005 \
  INPUT.SAMPLING_FRAME_NUM 3 \
  INPUT.SAMPLING_FRAME_WINDOE_NUM 3 \
  INPUT.SAMPLING_FRAME_VIDEO_NUM 3 \
  INPUT.LSJ_AUG.SQUARE_ENABLED False \
  TEST.EVAL_PERIOD 5000 \
  MODEL.WEIGHTS output/stage1/univs_r50_stage1.pth \
  OUTPUT_DIR output/v2/univs_r50_stage2