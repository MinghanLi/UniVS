# UniVS MODEL ZOO

## 1. Pretrained Models from [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md#panoptic-segmentation)
We use the pretrained models of Mask2Former trained on COCO panoptic segmentation datasets. 

If you want to train UniVS, please download them in `pretrained/m2f_panseg/` dir.

<table>
  <tr>
    <th>Backbone</th>
    <th>Model id</th>
    <th>Model</th>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>47430278_4</td>
    <td><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_R50_bs16_50ep/model_final_94dc52.pkl">model</a></td>
  </tr>
  <tr>
    <td>Swin-Tiny</td>
    <td>48558700_1</td>
    <td><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_tiny_bs16_50ep/model_final_9fd0ae.pkl">model</a></td>
  </tr>
  <tr>
    <td>Swin-Base</td>
    <td>48558700_7</td>
    <td><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl">model</a></td>
  </tr>
  <tr>
    <td>Swin-Large</td>
    <td>47429163_0</td>
    <td><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl">model</a></td>
  </tr>
</table>

## 2. UniVS Models
UniVS achieves superior performance on 10 benchmarks, using the same model with the same model parameters. UniVS has three training stages: image-level joint training, video-level joint training, and long video-level joint training. We provide all the checkpoints of all stages for models with different backbones. 

If you want to evaluate UniVS on different stages, please download them to `output/stage{1,2,3}/` dirs respectively.

### Stage 2: Video-level Joint Training
Note that the input image for Swin-Tiny/Base/Large backbones must have a shape of 1024 x 1024.
<table>
  <tr>
    <th>Backbone</th>
    <th>YAML</th>
    <th>Model</th>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>univs_r50_stage2</td>
    <td><a href="https://drive.google.com/file/d/1IX3HKIkZJKmA58VJiF9Xh0fJPYXLQ1Nc/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Tiny</td>
    <td>univs_swint_stage2</td>
    <td><a href="https://drive.google.com/file/d/1A48BoH1mlLYYcRuJFajoR_2iFscgnncU/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Base</td>
    <td>univs_swinb_stage2</td>
    <td><a href="https://drive.google.com/file/d/196YHDC01ghO34UL5RGFCqurPngR5EIOa/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Large</td>
    <td>univs_swinl_stage2</td>
    <td><a href="https://drive.google.com/file/d/1aIANl9LpzT3bsd90Kna8Zr08v5mvyfdz/view?usp=sharing">model</a></td>
  </tr>
</table>

### Stage 3: Long Video-level Joint Training 
All numbers reported in the paper uses the following models. This stage supports input images of all aspect ratios, and the results perform better when the short side is between 512 and 720. The trained models will be released soon.

<table>
  <tr>
    <th>Backbone</th>
    <th>YAML</th>
    <th>Model</th>

  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>univs_r50_stage3</td>
    <td><a href="?">-</a></td>
  </tr>
  <tr>
    <td>Swin-Tiny</td>
    <td>univs_swint_stage3</td>
    <td><a href="?">-</a></td>
  </tr>
  <tr>
    <td>Swin-Base</td>
    <td>univs_swinb_stage3</td>
    <td><a href="">-</a></td>
  </tr>
  <tr>
    <td>Swin-Large</td>
    <td>univs_swinl_stage3</td>
    <td><a href="">-</a></td>
  </tr>
</table>

