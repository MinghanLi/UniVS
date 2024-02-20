# [UniVS](?): Unified and Universal Video Segmentation with Prompts as Queries

[Minghan LI](https://scholar.google.com/citations?user=LhdBgMAAAAAJ), [Shuai LI](https://scholar.google.com/citations?hl=en&user=Bd73ldQAAAAJ), [Xindong Zhang](https://scholar.google.com/citations?hl=en&user=q76RnqIAAAAJ) and [Lei ZHANG](https://www4.comp.polyu.edu.hk/~cslzhang/)

[[arXiv]](?)

<div align="center">
  <img src="imgs/vs_tasks.jpg" width="90%" height="100%"/>
</div><br/>

## Updates
* **`Jan. 30, 2024`:** Paper is available now.


## Installation
See [installation instructions](INSTALL.md).

## Datasets
See [Datasets preparation](./datasets/README.md).

## Unified Training for Images and Videos
We provide a script `train_net.py`, that is made to train all the configs provided in UniVS.

Download [pretrained weights of Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) and save it into the path 'pretrained/m2f/*.pth', then run three stages one by one:
```
sh tools/run/univs_r50_stage1.sh
sh tools/run/univs_r50_stage2.sh
sh tools/run/univs_r50_stage3.sh
```

## Unified Inference for videos
Download [trained weights](), and save it into the path 'pretrained/univs/*.pth'. We support multiple ways to evaluate UniVS:
```
# test all six tasks using ResNet50 backbone (one-model)
$ sh tools/test/test_r50.sh

# test pvos only using ResNet50, swin-T/B/L backbones
$ sh tools/test/individual_task/test_pvos.sh
```

## <a name="CitingUniVS"></a>Citing UniVS

If you use UniVS in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@misc{li2023boxvis,
      title={BoxVIS: Video Instance Segmentation with Box Annotations}, 
      author={Minghan Li, Shuai Li and Lei Zhang},
      year={2024},
      eprint={2303.14618},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

Our code is largely based on [Detectron2](https://github.com/facebookresearch/detectron2), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [VITA](https://github.com/sukjunhwang/VITA), [ReferFormer](https://github.com/wjn922/ReferFormer), [SAM](https://github.com/facebookresearch/segment-anything) and [UniNEXT](https://github.com/MasterBin-IIAU/UNINEXT). We are truly grateful for their excellent work.
