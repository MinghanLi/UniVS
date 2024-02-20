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


if __name__ == '__main__':
    source_model = 'output/ytvis21/minvis_r50_ytvis21/model_0001999.pth'
    copy_TeacherNet_weights(source_model)

    exit()