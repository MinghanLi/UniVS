import os
import pickle
import argparse
import torch
from einops import repeat


def parse_args():
    parser = argparse.ArgumentParser("D2 model converter")
    parser.add_argument("--num_frames", default=2, type=int, help="Number of frames")
    parser.add_argument("--source_model", default="", type=str, help="Path or url to the  model to convert")
    return parser.parse_args()


def inflated_weights(source_model, n_heads=8, n_points=4, num_frames=3):
    if os.path.splitext(source_model)[-1] not in {".pth", ".pkl"}:
        raise ValueError("You should save weights as pth file")

    if source_model[-3:] == 'pkl':
        load_model = pickle.load(open(source_model, 'rb'))
    else:
        load_model = torch.load(source_model, map_location=torch.device('cpu'))
    source_weights = load_model["model"]
    keys = list(source_weights.keys())
    for k in keys:
        if k.startswith('sem_seg_head.pixel_decoder.transformer.encoder.layers'):
            if k.split('.')[-2] in {'sampling_offsets', 'attention_weights'}:
                D = 1 if k.split('.')[-2] == 'attention_weights' else 2
                if k.split('.')[-1] == 'bias':
                    source_weights[k] = repeat(source_weights[k], '(H L K D) -> (H T L K D)',
                                               H=n_heads, T=num_frames, K=n_points, D=D)
                elif k.split('.')[-1] == 'weight':
                    source_weights[k] = repeat(source_weights[k], '(H L K D) C -> (H T L K D) C',
                                               H=n_heads, T=num_frames, K=n_points, D=D)

    load_model["model"] = source_weights

    output_model = source_model[:-4] + '_inflated_to_f' + str(num_frames) + '.pth'
    print('Inflated weights are saved as:', output_model)
    torch.save(load_model, output_model)

    return output_model


if __name__ == "__main__":
    args = parse_args()
    inflated_weights(args.num_frames, args.source_model)