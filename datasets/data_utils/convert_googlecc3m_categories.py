import os
import json

concept_file = os.path.join('pretrained/regionclip/concept_emb/googlecc_nouns_6250.txt')
save_category_path = os.path.join('pretrained/regionclip/concept_emb/googlecc_nouns_6250.json')


def convert_googlecc_from_text():
    """
    Convert Nouns of GoogleCC 3M into category with ID, saved as Dict.
    Returns:

    """
    categories = []

    with open(concept_file, 'r') as f:
        for i, line in enumerate(f):
            concept = line.strip()
            concept = concept.split(',')[0] if ',' in concept else concept
            assert isinstance(concept, str), f'category name {concept} should be str.'
            categories.append({"id": i, "name": concept})

    with open(save_category_path, "w") as f:
        json.dump(categories, f)


def load_googlecc_category():
    return json.load(open(save_category_path, 'r'))


if __name__ == '__main__':
    convert_googlecc_from_text()

    load_googlecc_category()


