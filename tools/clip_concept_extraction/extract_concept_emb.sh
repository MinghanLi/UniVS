# extract text embeddings of combined classes in videos
CUDA_VISIBLE_DEVICES=0 python tools/clip_concept_extraction/extract_concept_emb/extract_concept_emb.py \
  --num-gpus 1 \
  --dist-url tcp://127.0.0.1:50153 \
  --config-file configs/clip/CLIP_lang_encoder_emb640_combined_datasets.yaml \
  OUTPUT_DIR "./datasets/concept_emb/"