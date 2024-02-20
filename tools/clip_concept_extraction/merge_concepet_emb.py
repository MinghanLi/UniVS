import os
import torch

root_dir = 'datasets/concept_emb'
input_path1 = 'combined_datasets_cls_emb_rn50x4.pth'
input_path2 = 'ytvis19_40_cls_emb_rn50x4.pth'
out_path = 'combined_datasets_cls_emb_rn50x4.pth'

emb1 = torch.load(os.path.join(root_dir, input_path1))
emb2 = torch.load(os.path.join(root_dir, input_path2))
out_emb = torch.cat([emb1, emb2])

print(emb1.shape, emb2.shape, out_emb.shape)
torch.save(out_emb, os.path.join(root_dir, out_path))