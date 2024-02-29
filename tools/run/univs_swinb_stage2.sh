CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
  --num-gpus 4 \
  --dist-url tcp://127.0.0.1:50159 \
  --config-file configs/univs/univs_swinb_stage2.yaml \
  --resume \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.00005 \
  INPUT.SAMPLING_FRAME_NUM 5 \
  INPUT.SAMPLING_FRAME_VIDEO_NUM 5 \
  TEST.EVAL_PERIOD 5000 \
  MODEL.BACKBONE.FREEZE_AT 4 \
  MODEL.WEIGHTS pretrained/univs/swinb/v1/univs_prompt_swinb_bs4_f4_video_2x_200queries_c1+univs+casa_psa_frozenbb_ft_refer+burst.pth \
  OUTPUT_DIR output/v2/univs_swinb_stage2