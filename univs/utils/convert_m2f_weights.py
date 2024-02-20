import os
import torch
import pickle


def copy_TeacherNet_weights(source_model):
    if os.path.splitext(source_model)[-1] not in {".pth", ".pkl"}:
        raise ValueError("You should save weights as pth file")

    if source_model[-3:] == 'pkl':
        load_model = pickle.load(open(source_model, 'rb'))
    else:
        load_model = torch.load(source_model, map_location=torch.device('cpu'))

    source_weights = load_model["model"]
    keys = list(source_weights.keys())

    for k in keys:
        # replace 'backbone' with 'backbone_t'
        if k.startswith('backbone.') or k.startswith('sem_seg_head.'):
            new_k = '.'.join([k.split('.')[0]+'_t'] + k.split('.')[1:])
            source_weights[new_k] = source_weights[k]

    load_model["model"] = source_weights

    output_model = source_model[:-4] + '_copy_teacher.pth'
    print('Copy weights for TeacherNet are saved as:', output_model)
    torch.save(load_model, output_model)

    return output_model

def convert_pkl_to_pth(source_model):
    if source_model[-3:] == 'pkl':
        load_model = pickle.load(open(source_model, 'rb'))
    
    source_weights = load_model["model"]
    keys = list(source_weights.keys())

    for k in keys:
        if "static_query" in k:
            new_k = k.replace("static_query", "query_feat")
            source_weights[new_k] = source_weights[k]
            source_weights.pop(k)
            print(f"replace {k} to {new_k}")

    output_model = source_model.replace('.pkl', '.pth')
    torch.save(load_model, output_model)
    print("Converted model are saved in", output_model)


if __name__ == '__main__':
    #source_model = 'pretrained/m2f_panseg/model_final_94dc52.pkl'
    source_model = 'pretrained/m2f_panseg/model_final_54b88a.pkl'
    # copy_TeacherNet_weights(source_model)
    convert_pkl_to_pth(source_model)

    exit()