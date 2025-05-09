# LSRL-Net: A Level Set-Guided Re-learning Network for Semi-supervised Cardiac and Prostate Segmentation
This repository is for our paper "LSRL-Net: A Level Set-Guided Re-learning Network for Semi-supervised Cardiac and Prostate Segmentation"

## Requirements
Some important required packages include:

* Pytorch version >=0.4.1.

* TensorBoardX

* Python == 3.7

* Efficientnet-Pytorch

* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy，Batchgenerators ......

## Usage

### 1、Clone the repo;
```
https://github.com/0xRiverr/LSRL-Net.git
```

### 2、Data Preparation;

The division method of training/validation/test set can be seen:

[ACDC dataset](https://github.com/0xRiverr/LSRL-Net/tree/main/data/ACDC)

[Prostate dataset](https://github.com/0xRiverr/LSRL-Net/tree/main/data/Prostate)

The data that can be used to train our code can be seen:

[ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)

[Prostate dataset](https://promise12.grand-challenge.org/)


The division of labeled/unlabeled datasets can be found in [this code](https://github.com/0xRiverr/LSRL-Net/blob/main/code/train_LP_ACDC.py)

You can regenerate the training data:
```
cd LSRL-Net/code/dataloaders

python acdc_data_processing.py
```

### 3、Level Set;

The code for pre-segmentation using the level set method can be found [here]()

### 4、Train the model;

```
cd LSRL-Net/code

CUDA_VISIBLE_DEVICES=0 python train_LP_ACDC.py --root_path ../data/ACDC --exp ACDC/LSRL-Net --num_classes 4 --labeled_num 3 --final_labeled_num 7
```
### 5、Test the model;
```
cd LSRL-Net/code

CUDA_VISIBLE_DEVICES=0 python test_2D_fully_acdc.py --root_path ../data/ACDC --exp ACDC/LSRL-Net --num_classes 4 --labeled_num 3 --final_labeled_num 7
```
Our code is based on the [UAMT](https://github.com/yulequan/UA-MT), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [SLC-Net](https://github.com/igip-liu/SLC-Net). Thanks for these authors for their valuable works.
