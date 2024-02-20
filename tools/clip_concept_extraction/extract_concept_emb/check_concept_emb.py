import os
import sys
from pathlib import Path

import argparse
import torch
import torch.nn.functional as F

file = Path(__file__).resolve()
parent = str(file.parent)
parent_new = '/'.join(parent.split('/')[:-2])

sys.path.append(parent_new)
file = Path(__file__).resolve()
parent = str(file.parent)

from videosam.data.datasets.coco_panoptic import COCO_PANOPTIC_CATEGORIES
from videosam.modeling.language.extract_class_embed import cat2ind


def parse_args():
    parser = argparse.ArgumentParser("check the similarity of concept embeddings")
    parser.add_argument("--cls_file",
                        default="datasets/concept_emb/lvis1203_ytvis40_ovis25_bdd8_obj365_coco133_ade150.txt",
                        type=str, help="the file name of storing concept strings (.txt format)")
    parser.add_argument("--cls_emb_file",
                        default="pretrained/regionclip/concept_emb/lvis1203_ytvis40_ovis25_bdd8_obj365_coco133_ade150_cls_emb_rn50x4.pth",
                        type=str, help="")
    parser.add_argument("--save_dir",
                        default="output/text_encoder/class_emb_analysis/",
                        type=str, help="")
    parser.add_argument("--merge_newly_cls_emb", default=False, type=bool, help="")
    parser.add_argument("--newly_cls_emb_file",
                        default="pretrained/regionclip/concept_emb/vipseg_123_cls_emb_rn50x4.pth",
                        type=str, help="")
    return parser.parse_args()


def read_concept_from_txt(args):
    # input concepts
    concept_file = os.path.join(parent_new, args.cls_file)

    concept_dict = {}
    i = 0
    with open(concept_file, 'r') as f:
        for line in f:
            concept_dict[i] = line.strip()
            i += 1

    return concept_dict


def get_topk_categories():
    cls_emb_path = os.path.join(parent_new, args.cls_emb_file)
    cls_emb = torch.load(cls_emb_path)
    cls_emb_norm = F.normalize(cls_emb, dim=-1)

    sim_map = cls_emb_norm @ cls_emb_norm.t()
    topk = 5
    topk_scores, topk_idxs = torch.topk(sim_map, k=topk, dim=-1)

    # read concept
    if args.cls_file is not None:
        cat_dict = read_concept_from_txt(args)
    else:
        cat_dict = cat2ind(COCO_PANOPTIC_CATEGORIES)

    assert len(cat_dict) == len(cls_emb)
    os.makedirs(args.save_dir, exist_ok=True)
    save_file = os.path.join(args.save_dir, args.cls_file.split('/')[-1])
    with open(save_file, "w") as text_file:
        for i, (topk_score, topk_idx) in enumerate(zip(topk_scores, topk_idxs)):
            topk_cate = [cat_dict[idx] for idx in topk_idx.tolist()]
            print(f"The topk{topk} closest of category '{cat_dict[i]}' is: {topk_cate}, with score {topk_score}",
                  file=text_file)


def merge_newly_category_to_combined_category():
    cls_emb_path = os.path.join(parent_new, args.cls_emb_file)
    cls_emb = torch.load(cls_emb_path)

    newly_cls_emb_path = os.path.join(parent_new, args.newly_cls_emb_file)
    newly_cls_emb = torch.load(newly_cls_emb_path)

    cls_emb = torch.cat([cls_emb, newly_cls_emb])
    newly_data_name = args.newly_cls_emb_file.split('/')[-1].replace('_cls_emb_rn50x4.pth', '')
    newly_data_name = ''.join(newly_data_name.split('_'))
    save_path = args.cls_emb_file.replace('cls_emb_rn50x4.pth', newly_data_name+'_cls_emb_rn50x4.pth')
    torch.save(cls_emb, save_path)


if __name__ == "__main__":
    args = parse_args()
    if args.merge_newly_cls_emb:
        merge_newly_category_to_combined_category()
    else:
        get_topk_categories()