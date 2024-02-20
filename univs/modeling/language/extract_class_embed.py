from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from videosam.data.datasets.ytvis import YTVIS_CATEGORIES_2019, YTVIS_CATEGORIES_2021
from videosam.data.datasets.ovis import OVIS_CATEGORIES
from videosam.data.datasets.bdd100k import BDD_TRACK_CATEGORIES
from videosam.data.datasets.burst import BURST_CATEGORIES
from videosam.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
from videosam.data.datasets.objects365_v2 import OBJECTS365_CATEGORIES
from videosam.data.datasets.ade20k_panoptic import ADE20K_150_CATEGORIES
from videosam.data.datasets.coco_panoptic import COCO_PANOPTIC_CATEGORIES

from .clip_prompt_utils import pre_tokenize, clean_strings


def cat2ind(categories):
    ind_to_class = {}
    if isinstance(categories, str):
        ind_to_class[0] = categories

    elif isinstance(categories, list):
        if isinstance(categories[0], dict):
            for index, x in enumerate(categories):
                assert "name" in x
                ind_to_class[index] = x["name"]
        elif isinstance(categories[0], str):
            for index, x in enumerate(categories):
                ind_to_class[index] = x
        else:
            raise ValueError("Only supported categories in {name: 'person'} or 'person'")

    else:
        raise ValueError("Only supported categories in a single list or a single str")

    return ind_to_class


def create_queries_and_maps(categories):
    ind_to_class = cat2ind(categories)
    label_name_list = [clean_strings(x) for index, x in ind_to_class.items()]
    labels = list(range(1, len(label_name_list) + 1))  # start from 1
    # len(categories) x 81 x 77, where 81 is the number of templates
    cate_tokensizers = pre_tokenize(label_name_list)

    return cate_tokensizers, labels


def extract_class_emb(test_categories=None, dataset_name=None, is_train=False):
    assert test_categories is not None or dataset_name is not None

    if test_categories is not None:
        if isinstance(test_categories, str):
            # input as .txt file
            test_categories_list = []
            if test_categories[-4:] == ".txt":
                with open(test_categories, 'r') as f:
                    for line in f:
                        test_categories_list.append(line.strip())
            else:
                test_categories_list = [test_categories]
        else:
            test_categories_list = test_categories

        # for example, test_categories = [{"name": "person"}]
        ind_to_class = cat2ind(test_categories_list)
        prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(test_categories_list)
        return {
            "ind_to_class": ind_to_class,
            "class_tokens": prompt_tokens if not is_train else None,
            "map_label_to_token": positive_map_label_to_tokens if not is_train else None,
        }

    else:
        
        if dataset_name.startswith("coco"):
            if "panoptic" in dataset_name:
                ind_to_class = cat2ind(COCO_PANOPTIC_CATEGORIES)
                if not is_train:
                    prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(COCO_PANOPTIC_CATEGORIES)
            else:
                ind_to_class = cat2ind(COCO_CATEGORIES)
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(COCO_CATEGORIES)
        elif dataset_name.startswith("lvis"):
            ind_to_class = cat2ind(LVIS_CATEGORIES)
            if not is_train:
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(LVIS_CATEGORIES)
        elif dataset_name.startswith("ade"):
            ind_to_class = cat2ind(ADE20K_150_CATEGORIES)
            if not is_train:
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(ADE20K_150_CATEGORIES)
        elif dataset_name.startswith("objects365"):
            ind_to_class = cat2ind(OBJECTS365_CATEGORIES)
            if not is_train:
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(OBJECTS365_CATEGORIES)
        elif dataset_name.startswith("ytvis_2019"):
            ind_to_class = cat2ind(YTVIS_CATEGORIES_2019)
            if not is_train:
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(YTVIS_CATEGORIES_2019)
        elif dataset_name.startswith("ytvis_2021"):
            ind_to_class = cat2ind(YTVIS_CATEGORIES_2021)
            if not is_train:
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(YTVIS_CATEGORIES_2021)
        elif dataset_name.startswith("ovis"):
            ind_to_class = cat2ind(OVIS_CATEGORIES)
            if not is_train:
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(OVIS_CATEGORIES)
        elif dataset_name.startswith("mots_bdd_box_track") or dataset_name.startswith("mots_bdd_seg_track"):
            ind_to_class = cat2ind(BDD_TRACK_CATEGORIES)
            if not is_train:
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(BDD_TRACK_CATEGORIES)
        elif dataset_name.startswith("mots_burst"):
            ind_to_class = cat2ind(BURST_CATEGORIES)
            if not is_train:
                prompt_tokens, positive_map_label_to_tokens = create_queries_and_maps(BURST_CATEGORIES)
        else:
            print(f"Unsupported dataset_name: {dataset_name}")
            return 

        return {
            "dataset_name": dataset_name,
            "ind_to_class": ind_to_class,
            "class_tokens": prompt_tokens if not is_train else None,
            "map_label_to_token": positive_map_label_to_tokens if not is_train else None,
        }

