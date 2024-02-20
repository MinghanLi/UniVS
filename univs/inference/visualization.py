import os
import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer


Colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'black', 'purple', 'orange', 'grey',
          'lime', 'pink', 'navy', 'gold', 'olive', 'chocolate', 'skyblue', 'brown']


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

class visualization_query_embds:
    def __init__(
        self,
        reduced_type='original', 
        t_components=2, 
        output_dir='output/visualization_query_embds/',
        apply_cls_thres=0.1,
    ):
        self.reduced_type = reduced_type
        self.t_components = t_components
        self.output_dir = output_dir
        
        self.apply_cls_thres = apply_cls_thres
    
    def visualization_query_embds(self, targets, reduced_type=None):
        reduced_type = reduced_type if reduced_type is not None else self.reduced_type
        output_dir = os.path.join(self.output_dir, reduced_type)
        os.makedirs(output_dir, exist_ok=True)

        for targets_per_video in targets:
            task = targets_per_video['task']
            video_name = targets_per_video["file_names"][0].split('/')[-2]

            embds = targets_per_video['embds']  # N_pred, T_prev, C

            if task in {'detection'}:
                logits = targets_per_video['logits'] # N_pred, T_prev, K
                scores = logits.mean(1).max(-1)[0]
                scores, sorted_idxs = scores.sort(descending=True)
                scores = scores[:len(Colors)]
                sorted_idxs = sorted_idxs[:len(Colors)]
                embds = embds[sorted_idxs]
                logits = logits[sorted_idxs]

                valid = scores >= self.apply_cls_thres
                embds = embds[valid]
                logits = logits[valid]
            else:
                logits = None

            if embds.nelement() == 0:
                continue

            embds = embds / torch.norm(embds, dim=-1, keepdim=True).clamp(min=1e-3)
            embds = embds.clamp(min=0.)
            embds = embds / torch.max(embds, dim=-1, keepdim=True)[0].clamp(min=1e-3)

            if reduced_type == 'original':
                self.visualization_query_embds_original(embds, video_name, output_dir)
            elif reduced_type == 'pca':
                self.visualization_query_embds_PCA(embds, video_name, output_dir)
            elif reduced_type == 'tsne':
                self.visualization_query_embds_TSNE(embds, logits, video_name, output_dir)

    def visualization_query_embds_PCA(self, data, video_name, output_dir):
        N, T, C = data.shape
        data = data.flatten(0, -2).cpu().numpy()

        c = sum([[Colors[i]]*T for i in range(N)], [])

        vid_id = video_name + '.jpg'
        output_path = os.path.join(output_dir, vid_id)
        print('Save PCA visualization of query embds in:', output_path)

        pca = PCA(n_components=3)
        reduced_data_tsne = pca.fit_transform(data)
        
        if self.t_components == 2:
            # plotting the first two principle components in 2D
            plt.figure()
            plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=c)
            plt.title('PCA in 2D')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.clf()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], reduced_data_tsne[:, 2])
            plt.title('PCA in 3D')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.clf()

    def visualization_query_embds_TSNE(self, data, video_name, output_dir):
        '''
        visualize query embeddings predicted from the entire video by t-SNE (t-distributed stochastic neighbor embeddings)
        data: N x T x C, where N is the number of objects, and T is the number of frames in the video
        '''
        N, T, C = data.shape
        data = data.flatten(0, -2).cpu().numpy()

        c = sum([[Colors[i]]*T for i in range(N)], [])

        vid_id = video_name + '.jpg' 
        output_path = os.path.join(output_dir, vid_id)
        print('Save TSNE visualization of query embds in:', output_path)

        if self.t_components == 2:
            tsne = TSNE(n_components=2)
            reduced_data_tsne = tsne.fit_transform(data)
            
            plt.figure()
            plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=c)
            plt.title('t-SNE in 2D')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.clf()

        elif self.t_components == 3:
            tsne_3d = TSNE(n_components=3)
            reduced_data_tsne = tsne_3d.fit_transform(data)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], reduced_data_tsne[:, 2])
            plt.title('t-SNE in 3D')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.clf()
        
        else:
            raise ValueError

    def visualization_query_embds_original(self, pred_embds, pred_logits, video_name, output_dir):
        for i, (embds, logits) in enumerate(zip(pred_embds, pred_logits)):
            s, l = logits.mean(0).max(-1)
            if s < self.apply_cls_thres:
                continue

            embds_np = embds.t().cpu().numpy()

            save_by_category = True
            if save_by_category:
                output_dir = output_dir + '/category/' + str(int(l))
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, video_name + '_' + str(i) + '.jpg')
            else:
                output_dir = output_dir + '/video/' + video_name
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, str(i) + '.jpg')

            im = plt.imshow(embds_np, cmap='jet', vmin=0, vmax=1)
            cbar = plt.colorbar(im, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.title(str(int(l))+' + %.2f' % s)
            plt.savefig(output_path, dpi=300,bbox_inches='tight')
            plt.clf()


