CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py \
  --num-gpus 8 \
  --dist-url tcp://127.0.0.1:50159 \
  --config-file configs/univs/univs_swint_stage2.yaml \
  --resume \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.BASE_LR 0.00005 \
  INPUT.SAMPLING_FRAME_NUM 5 \
  INPUT.SAMPLING_FRAME_VIDEO_NUM 5 \
  TEST.EVAL_PERIOD 5000 \
  MODEL.BACKBONE.FREEZE_AT 4 \
  MODEL.WEIGHTS output/stage1/univs_swint_stage1.pth \
  OUTPUT_DIR output/v2/univs_swint_stage2