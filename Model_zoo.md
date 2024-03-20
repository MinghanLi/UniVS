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

## 2. Prepare CLIP Text Encoder & Category Embeddings
### 2.1 Download pretrained model of CLIP Text Encoder from [Google drive](https://drive.google.com/file/d/1jzjiZHRag63KSXZq3XpLXj5FFDRLL9ZX/view?usp=sharing) and put it in the dir `'pretrained/regionclip/regionclip/'`


### 2.2 Prepare Category Embeddings for UniVS
a) Extract all category names that existed in the following datasets, please refer to `datasets/concept_emb/combined_datasets.txt` and `datasets/concept_emb/combined_datasets_category_info.py`

b) Extract concept embeddings of category names from CLIP Text Encoder. For your convenience, you can directly download the converted file from [Google drive](https://drive.google.com/file/d/1Wmw6n_u7NB3lARjXp6oUe9VsGa332zi8/view?usp=sharing) and put it in the dir `'datasets/concept_emb/'` .

c) Alternatively, you can run the code to generate in your server
```
$ cd UniVS
$ sh tools/clip_concept_extraction/extract_concept_emb.sh
```

## 3. UniVS Models
UniVS achieves superior performance on 10 benchmarks, using the same model with the same model parameters. UniVS has three training stages: image-level joint training, video-level joint training, and long video-level joint training. We provide all the checkpoints of all stages for models with different backbones. 

If you want to evaluate UniVS on different stages, please download them to `'output/stage{1,2,3}/'` dirs respectively.

### Stage 2: Video-level Joint Training
Note that the input image for Swin-Tiny/Base/Large backbones must have a shape of 1024 x 1024.
<table>
  <tr>
    <th>Backbone</th>
    <th>YAML</th>
    <th>INPUT</th>
    <th>Model</th>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>univs_r50_stage2</td>
    <td> Any resolution</td>
    <td><a href="https://drive.google.com/file/d/1IX3HKIkZJKmA58VJiF9Xh0fJPYXLQ1Nc/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Tiny</td>
    <td>univs_swint_stage2</td>
    <td> 1024 * 1024</td>
    <td><a href="https://drive.google.com/file/d/1A48BoH1mlLYYcRuJFajoR_2iFscgnncU/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Base</td>
    <td>univs_swinb_stage2</td>
    <td> 1024 * 1024</td>
    <td><a href="https://drive.google.com/file/d/196YHDC01ghO34UL5RGFCqurPngR5EIOa/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Large</td>
    <td>univs_swinl_stage2</td>
    <td> 1024 * 1024</td>
    <td><a href="https://drive.google.com/file/d/1aIANl9LpzT3bsd90Kna8Zr08v5mvyfdz/view?usp=sharing">model</a></td>
  </tr>
</table>

### Stage 3: Long Video-level Joint Training 
All numbers reported in the paper uses the following models. This stage supports input images of all aspect ratios, and the results perform better when the short side is between 512 and 720. 

<table>
  <tr>
    <th>Backbone</th>
    <th>YAML</th>
    <th>Input</th>
    <th>Model</th>

  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>univs_r50_stage3</td>
    <td> Any resolution</td>
    <td><a href="https://drive.google.com/file/d/1PJQ7ryyPiK4-YagBxMCKNkm5C_nLjhv3/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Tiny</td>
    <td>univs_swint_stage3</td>
    <td> Any resolution</td>
    <td><a href="https://drive.google.com/file/d/1yzL_uUETd_qGhkmUOsU59X1ItFVu8JKq/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Base</td>
    <td>univs_swinb_stage3</td>
    <td> Any resolution</td>
    <td><a href="https://drive.google.com/file/d/19Y_icBnyOh5TC-BS7mJdoQn0lN80jTOv/view?usp=sharing">model</a></td>
  </tr>
  <tr>
    <td>Swin-Large</td>
    <td>univs_swinl_stage3</td>
    <td> Any resolution</td>
    <td><a href="https://drive.google.com/file/d/1RujQUMrl46VktO4SmMvbmBEnr1ZKjXd-/view?usp=sharing">model</a></td>
  </tr>
</table>


## 4. Inference UniVS on development set 

If the ground-truth for the validation set has already been released, as in the case of the DAVIS and VSPW benchmarks, we will use the first 20% of the data from the validation set for the development set to enable rapid inference. If the ground-truth for the validation set has not been released, such as YouTube-VIS benchmark, we will divide the training set into two parts: training (train_sub.json) and development (valid_sub.json) sets. In this scenario, the data in the development set will not be seen during the training phase.

For your convenience in evaluation, we provide the converted development annotation files, which you can download from [here](https://drive.google.com/file/d/1UFhkdcMz_HIBYEQSFy72qpC2byZkCAvC/view?usp=sharing). After downloading, please unzip the file and store it according to the following structure:

```
datasets/
  |---ytvis_2021/
    |---train.json  
    |---train_sub.json  (90% videos in training set)
    |---valid_sub.json  (10% videos in training set)
    |---train/
      |---JPEGImages/
    |---valid/
      |---JPEGImages/

  |---ovis/
    |---train.json  
    |---train_sub.json  (90% videos in training set)
    |---valid_sub.json  (10% videos in training set)
    |---train/
      |---JPEGImages/
    |---valid/
      |---JPEGImages/

  |---VSPW_480p/
      |---val_cocovid.json
      |---dev_cocovid.json  (first 50 videos in val set, only for debug)
      |---data/

  |---vipseg/
    |---VIPSeg_720P/
      |--- panoptic_gt_VIPSeg_val_cocovid.json
      |--- panoptic_gt_VIPSeg_val_sub_cocovid.json
      |--- imgs/
      |--- panomasksRGB/

  |---DAVIS/
    2017_val.json
    |---JPEGImages/
      |---Full-Resolution/

  |---viposeg/
    |---valid/
      |---valid_cocovid.json
      |---dev_cocovid.json 
      |---JPEGImages/

  |---ref-davis/
    |---valid_0.json
    |---valid_1.json
    |---valid_2.json
    |---valid_3.json
    |---valid/
      |---JPEGImages/
```

The results evaluated on the development sets are presented below. You can obtain these results by running the test scripts located in the `tools/test/` directory.

Note that the results from this section are solely applicable for code debugging and **should not** be used for performance comparison with other methods in the paper.

<div align="center">
  <img src="imgs/stage3_dev_results.png" width="100%" height="100%"/>
</div><br/>