
# 1. Prepare Category Embeddings for UniVS
a) Extract all category names that existed in the following datasets, please refer to `datasets/concept_emb/combined_datasets.txt` and `datasets/concept_emb/combined_datasets_category_info.py`

b) Extract concept embeddings of category names from CLIP Text Encoder. For your convenience, you can directly download the converted file from [Google drive](https://drive.google.com/file/d/1Wmw6n_u7NB3lARjXp6oUe9VsGa332zi8/view?usp=sharing) and put it in the dir `'datasets/concept_emb/'`.

c) Alternatively, you can run the code to generate in your server
```
$ cd UniVS
$ sh tools/clip_concept_extraction/extract_concept_emb.sh
```

# 2. Prepare Datasets for UniVS
Data preparation follows [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT/tree/master/conversion). Thanks a lot :) 

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them. All register datasets can be founded in `univs/data/datasets/builtin.py`.


The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  burst/
  coco/
  DAVIS/
  entityseg/
  lvis/
  mose/
  ovis/
  refcoco/
  ref-davis/  # only inference
  sa_1b/
  viposeg/    # only inference
  vipseg/
  VSPW_480p/  # only inference
  ytbvos/
  ytvis_2019/ 
  ytvis_2021/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`. If left unset, the default is `./datasets` relative to your current working directory.

## Expected dataset structure for [SA-1B](https://ai.facebook.com/datasets/segment-anything-downloads/):
a) You can use [SA-1B-Downloader](https://github.com/KKallidromitis/SA-1B-Downloader) to download it (11M images in total, but a subset is enough)
```
$ cd datasets
$ ln -s /path/to/your/sa_1b/dataset 
```

```
sa_1b/
    images/
    annotations/
    annotations_250k/
        {annotations_250k_*}.json
```

b) Split SA-1B images into several sub-files for dataloader, named as annotations_250k_*.json
```
python datasets/data_utils/split_sa1b_dataset.py
```

## Expected dataset structure for [COCO](https://competitions.codalab.org/competitions/20128) and [LVIS](https://www.lvisdataset.org/dataset):

a) Download images and annotations for [COCO](https://competitions.codalab.org/competitions/20128)

b) LVIS uses the COCO 2017 images, and you only need to download annotations [here](https://www.lvisdataset.org/dataset).

```
coco/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
    panoptic_train2017.json
    panoptic_train2017_cocovid.json (converted)
    panoptic_train2017/

# only use images higher than 512p (short edge)
python datasets/data_utils/convert_lvis_to_cocovid.py

lvis/
  lvis_v1_train.json
  lvis_v1_train_video512p.json (converted)
  lvis_v1_val.json
```

c) Convert COCO panoptic annotations into cocovid format
```
$ python datasets/data_utils/convert_coco_pan_seg_to_cocovid_train.py
```

## Expected dataset structure for [RefCOCO]()
a) Download processed json files by [SeqTR](https://github.com/seanzhuh/SeqTR) from [Google Drive](https://drive.google.com/drive/folders/1IXnSieVr5CHF2pVJpj0DlwC6R3SbfolU). We need three folders: refcoco-unc, refcocog-umd, and refcocoplus-unc. These folders should be organized as below.
```
refcoco/
  refcoco/
    instances_refcoco_{train,val,testA,testB}.json
    instances.json
  refcoco+/
    instances_refcoco+_{train,val,testA,testB}.json
    instances.json
  refcocog/
    instances_refcocoG_{train,val,test}.json
    instances.json
```

b) Convert annotations to cocovid format
```
$ python datasets/data_utils/convert_refcoco_to_cocovif_1.py
$ python datasets/data_utils/convert_refcoco_to_cocovif_2.py
$ python datasets/data_utils/convert_refcoco_to_cocovif_3.py
```

## Expected dataset structure for [EntitySeg-v1.0](https://github.com/adobe-research/EntitySeg-Dataset)
a) Download images from [Google drive](https://drive.google.com/drive/folders/1yX2rhOroyhUCGCrmzSm7DL4BfQWvcG0v) or [Hugging face](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg)

b) Download annotations from [Github](https://github.com/adobe-research/EntitySeg-Dataset/releases/tag/v1.0)

c) Unzip images and annotations, we use panoptic segmentation here

d) Convert to cocovid format
```
$ python datasets/data_utils/convert_entityseg_inst_seg_to_cocovid_train.py
$ python datasets/data_utils/convert_entityseg_pan_seg_to_cocovid_train.py
```

e) the data format
```
entityseg/
  annotations/
    entityseg_insseg_train_cocovid.json  
    entityseg_panseg_train_cocovid.json
    entityseg_train_{01, 02, 03}.json
  images/
```

## Expected dataset structure for [YouTubeVIS 2021](https://competitions.codalab.org/competitions/28988) or [Occluded VIS](http://songbai.site/ovis/):
a) Only need to download images and annotations and put them into the path 
```
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
      *.jpg
```

```
ovis/
  {train,valid,test}.json
  {train,valid,test}/
    JPEGImages/
```

## Expected dataset structure for [YouTubeVOS](https://youtube-vos.org/dataset/) and [Ref-YouTubeVOS](https://drive.google.com/drive/folders/1J45ubR8Y24wQ6dzKOTkfpd9GS_F9A2kb):
a) Download images and annotations of original [YTVOS-2018 dataset](https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f)

b) Download RefVOS annotations to get [meta-expression](https://drive.google.com/drive/folders/1J45ubR8Y24wQ6dzKOTkfpd9GS_F9A2kb)

c) Convert annotaions to cocovid format, and the data should look like

```
$ python datasets/data_utils/convert_ytvos_to_cocovid_train.py
$ python datasets/data_utils/convert_ytvos_to_cocovid_val.py
$ python datasets/data_utils/convert_refytvos_to_cocovid_train.py
$ python datasets/data_utils/convert_refytvos_to_cocovid_val.py

ytbvos/
    train/
        JPEGImages/
        Annotations/
    val/
        JPEGImges/
        Annotations/
    meta_expressions/ (for refytbvos)
        train/
            meta_expressions.json
        val/
            meta_expressions.json
        test/
            meta_expression.json
    meta.json
    train.json (after convert, for ytbvos)
    valid.json (after convert, for ytbvos)
    train_ref.json (after convert)
    valid_ref.json (after convert)
```

## Expected dataset structure for [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/tree/main), [VSPW](https://github.com/VSPW-dataset/VSPW-dataset-download) and [VIPOSeg](https://github.com/yoxu515/VIPOSeg-Benchmark):
Note that VIPSeg is used for training, while VSPW and VIPOSeg are used for inference only.

a) Download the VIPSeg from the official [Github]((https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/tree/main)) and change images to 720P
cd datasets/vipseg

```
$ python change2_720p.py
```

b) Convert it to cocovid format with original resolution
```
# original resolution for training   
$ python datasets/data_utils/convert_vipseg_to_cocovid.py  
```
c) Convert it to cocovid format with 720p
```
# 720p resolution for standard inference
$ python datasets/data_utils/convert_vipseg720p_to_cocovid.py  
```

d) Data format of VIPSeg dataset
```
vipseg/
  imgs/
  panomasksRGB/  
  panoptic_gt_VIPSeg_train_cocovid.json  # original resolution
  panoptic_gt_VIPSeg_val_cocovid.json    # original resolution
  panoptic_gt_VIPSeg_test_cocovid.json   # original resolution
  VIPSeg_720P/
    imgs/
    panomasks/
    panomasksRGB/
    panoptic_gt_VIPSeg_train_cocovid.json  # 720p resolution
    panoptic_gt_VIPSeg_val_cocovid.json    # 720p resolution
    panoptic_gt_VIPSeg_test_cocovid.json   # 720p resolution
```

e) Download datasets from offical websits for [VSPW](https://github.com/VSPW-dataset/VSPW-dataset-download) and [VIPOSeg](https://github.com/yoxu515/VIPOSeg-Benchmark), and convert them for evaluation
```
$ python datasets/data_utils/convert_vspw_to_cocovid_val.py
$ python datasets/data_utils/convert_vspw_to_cocovid_dev.py
$ python datasets/data_utils/convert_viposeg_to_cocovid_val.py
```



## Expected dataset structure for [TAO](https://taodataset.org/) and [BURST](https://github.com/Ali2500/BURST-benchmark):
TAO is a federated dataset for Tracking Any Object, containing 2,907 high resolution videos, captured in diverse environments, which are half a minute long on average. [BURST](https://github.com/Ali2500/BURST-benchmark) recently annotates the instance segmentations. 
```
# a) Move the images from TAO dataset to BURST dataset
$ cd datasets/
$ ln -s /path/to/datasets/TAO /path/to/datasets/BURST 

# b) Download the segmentation [annotations](https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip)

# c) Convert datasets 
$ python datasets/data_utils/convert_burst_to_cocovid_*.py 

# d) the data format
burst/
   frames/
     train/
     val/
   annotations/
     train/
     val/
     test/
     info/
     train_uni.json (converted)
     val_first_frame_uni.json (converted)
```

## Expected dataset structure for [MOSE](https://github.com/henghuiding/MOSE-api)

```
# a) Download data (train.tar.gz)
$ gdown 'https://drive.google.com/uc?id=10HYO-CJTaITalhzl_Zbz_Qpesh8F3gZR'

# b) Convert annotations to cocovid format
$ python datasets/data_utils/convert_mose_to_cocovid_train.py
$ python datasets/data_utils/convert_mose_to_cocovid_val.py
```


## Expected dataset structure for [Ref-DAVIS17](https://davischallenge.org/davis2017/code.html)

a) Downlaod the DAVIS2017 dataset from the [website](https://davischallenge.org/davis2017/code.html). Note that you only need to download the two zip files `DAVIS-2017-Unsupervised-trainval-480p.zip` and `DAVIS-2017_semantics-480p.zip`.

b) Download the text annotations from the [website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions) and put the zip files in the directory as follows

```
ref-davis
├── DAVIS-2017_semantics-480p.zip
├── DAVIS-2017-Unsupervised-trainval-480p.zip
├── davis_text_annotations.zip
```

c) Unzip these zip files
```
$ cd datasets/ref-davis
$ unzip -o davis_text_annotations.zip
$ unzip -o DAVIS-2017_semantics-480p.zip
$ unzip -o DAVIS-2017-Unsupervised-trainval-480p.zip
```

d) Preprocess the dataset to Ref-Youtube-VOS format 

```
$ cd ../../  # back to the main directory
$ python datasets/data_utils/convert_davis_to_ytvos.py
```

e) Finally, unzip the file `DAVIS-2017-Unsupervised-trainval-480p.zip` again (since we use `mv` in preprocess for efficiency).

```
$ unzip -o DAVIS-2017-Unsupervised-trainval-480p.zip
```

# 3. Add a New Dataset of Videos with *.mp4

```
# Convert original videos with .mp4 format to COCO annotations
$ python datasets/data_utils/convert_videos_to_coco_test.py --video_dir /path/to/your/videos --out_json /path/to/your/output/json

# Register the new dataset
$ vim univs/data/datasets/builtin.py
# Add the corresponding "dataset_name: (video_root, annotation_file), evaluator_type" into _PREDEFINED_SPLITS_RAW_VIDEOS_TEST
_PREDEFINED_SPLITS_RAW_VIDEOS_TEST = {
    # dataset_name: (video_root, annotation_file), evaluator_type
    "internvid-flt-1": ("internvid/raw/InternVId-FLT_1", "internvid/raw/InternVId-FLT_1.json", "none"),
}

```