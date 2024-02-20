import pickle
import torch


def convert_lang_encoder_weights(regionclip_model):
    # convert concept embeddings
    if regionclip_model[-3:] == 'pkl':
        load_regionclip_model = pickle.load(open(regionclip_model, 'rb'))
    else:
        load_regionclip_model = torch.load(regionclip_model, map_location=torch.device('cpu'))

    if "model" in load_regionclip_model:
        include_model = True
        regionclip_model_weights = load_regionclip_model["model"]
    else:
        include_model = False
        regionclip_model_weights = load_regionclip_model

    ref_keys = list(regionclip_model_weights.keys())
    for k in ref_keys:
        if k.startswith('lang_encoder'):
            new_k = '.'.join(k.split('.')[1:])
            regionclip_model_weights[new_k] = regionclip_model_weights[k]

        regionclip_model_weights.pop(k)

    if include_model:
        load_regionclip_model["model"] = regionclip_model_weights
    else:
        load_regionclip_model = regionclip_model_weights

    output_model = regionclip_model[:-4] + '_only_lang_encoder.pth'
    print('Merged RegionCLIP weights are saved in:', output_model)
    torch.save(load_regionclip_model, output_model)

    return output_model