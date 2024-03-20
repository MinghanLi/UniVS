CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --num-gpus 4 \
  --dist-url tcp://127.0.0.1:50159 \
  --config-file configs/univs/univs_swint_stage3.yaml \
  --resume \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.000025 \
  INPUT.SAMPLING_FRAME_NUM 5 \
  INPUT.SAMPLING_FRAME_WINDOE_NUM 5 \
  INPUT.SAMPLING_FRAME_VIDEO_NUM 7 \
  INPUT.LSJ_AUG.SQUARE_ENABLED False \
  TEST.EVAL_PERIOD 0 \
  SOLVER.STEPS '(81000, )' \
  SOLVER.MAX_ITER 89000 \
  MODEL.BoxVIS.EMA_ENABLED True \
  MODEL.WEIGHTS output/stage2/univs_swint_stage2.pth \
  OUTPUT_DIR output/v2/univs_swint_stage3_f7_wosquare_ema