import os
import torch
import torch.nn.functional as F

from detectron2.utils.visualizer import Visualizer

def display_instance_masks(inputs_per_image, metadata, results_instance, task_type='detection', output_dir='output', dataset_name=''):
        # masks_pred after sigmoid: N, H, W
        img_name = inputs_per_image["file_names"][0].split('/')[-1]
        img_id = img_name[:-4]

        # masks_pred after sigmoid: N, H, W
        if task_type == "grounding":
            exp_id = inputs_per_image["exp_id"]
            save_dir = os.path.join(output_dir, 'visual', dataset_name,
                                    img_id + '_' + exp_id)
        else:
            save_dir = os.path.join(output_dir, 'visual', dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        N_objs, H, W = results_instance.pred_masks.shape
        for k in list(results_instance._fields.keys()):
            if k in {"pred_boxes"}:
                results_instance._fields[k] = results_instance._fields[k].to('cpu')
            else:
                results_instance._fields[k] = results_instance._fields[k].detach().cpu()

        img = inputs_per_image["image"][0]
        img = F.interpolate(
            img[None].float(),
            (H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).long()
        img = img.permute(1, 2, 0).cpu().to(torch.uint8)

        save_path = '/'.join([save_dir, img_name])
        visualizer = Visualizer(img, metadata=metadata)
        VisImage = visualizer.draw_instance_predictions(results_instance)
        VisImage.save(save_path)

        print("Predicted instance masks are saved in:", save_path)