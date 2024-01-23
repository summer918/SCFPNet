## Installation

Our codebase is built upon [detectron2](https://github.com/facebookresearch/detectron2). You only need to install [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) following their instructions.

Please note that we used detectron 0.2.1 in this project. Higher versions of detectron might report errors.

## Data Preparation

- We evaluate our model on two FSOD benchmarks PASCAL VOC and MSCOCO following the previous work [TFA](https://github.com/ucbdrive/few-shot-object-detection).
- Please prepare the original PASCAL VOC and MSCOCO datasets and also the few-shot datasets following [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md) in the folder ./datasets/coco and ./datasets/pascal_voc respectively.
- Please run the scripts in ./datasets/coco and ./datasets/pascal_voc step by step to generate the support images for both many-shot base classes (used during meta-training) and few-shot classes (used during few-shot fine-tuning).

## Converting ImageNet pre-trained [PVT](https://github.com/whai362/PVT/tree/v2/classification#model-zoo) models into C4-based detection format
The script is 
```
python build_pvt_C4_from_official_model.py
```
We use the converted model pvt_v2_b2_li_C4.pth by default next.

## Model training and evaluation on MSCOCO

- We have three steps for model training, first pre-training the single-branch based model over base classes, then training the two-branch based model over base classes, and finally fine-tuning the two-branch based model over novel classes.
- The training script for pre-training the single-branch based model over base classes is
```
sh scripts/single_branch_pretraining_coco_pvt_v2_b2_li.sh
```
- Then initailized with the first step trained model, the script for training the two-branch based model over base classes is
```
sh scripts/two_branch_training_coco_pvt_v2_b2_li.sh
```
- Finally we perform 10/30-shot fine-tuning over novel classes, using the exact same few-shot datasets as [TFA](https://github.com/ucbdrive/few-shot-object-detection). The training script is
```
sh scripts/two_branch_few_shot_finetuning_coco_pvt_v2_b2_li.sh
```

## Model training and evaluation on PASCAL VOC

- We evaluate our model on the three splits as [TFA](https://github.com/ucbdrive/few-shot-object-detection).
- Similar as MSCOCO, we have three steps for model training.
- The training scripts for VOC split1 is 
```
sh scripts/single_branch_pretraining_pascalvoc_split1_pvt_v2_b2_li.sh
sh scripts/two_branch_training_pascalvoc_split1_pvt_v2_b2_li.sh
sh scripts/two_branch_few_shot_finetuning_pascalvoc_split1_pvt_v2_b2_li.sh
```
- The training scripts for VOC split2 is 
```
sh scripts/single_branch_pretraining_pascalvoc_split2_pvt_v2_b2_li.sh
sh scripts/two_branch_training_pascalvoc_split2_pvt_v2_b2_li.sh
sh scripts/two_branch_few_shot_finetuning_pascalvoc_split2_pvt_v2_b2_li.sh
```
- The training scripts for VOC split3 is 
```
sh scripts/single_branch_pretraining_pascalvoc_split3_pvt_v2_b2_li.sh
sh scripts/two_branch_training_pascalvoc_split3_pvt_v2_b2_li.sh
sh scripts/two_branch_few_shot_finetuning_pascalvoc_split3_pvt_v2_b2_li.sh
```
