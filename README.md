# IMEC-ZSDE

## Official code of Imbuing, Enrichment and Calibration: Leveraging Language for Unseen Domain Expansion
![image](https://github.com/LanchJL/IMEC-ZSDE/blob/main/IMG/fig_seg.jpg)
## Dataset
To download and use the data set, please refer to [PODA](https://github.com/astra-vision/PODA):
* **CITYSCAPES**: Follow the instructions in [Cityscapes](https://www.cityscapes-dataset.com/)
  to download the images and semantic segmentation ground-truths. Please follow the dataset directory structure:
  ```html
  <CITYSCAPES_DIR>/             % Cityscapes dataset root
  ├── leftImg8bit/              % input image (leftImg8bit_trainvaltest.zip)
  └── gtFine/                   % semantic segmentation labels (gtFine_trainvaltest.zip)
  ```

* **ACDC**: Download ACDC images and ground truths from [ACDC](https://acdc.vision.ee.ethz.ch/download). Please follow the dataset directory structure:
  ```html
  <ACDC_DIR>/                   % ACDC dataset root
  ├── rbg_anon/                 % input image (rgb_anon_trainvaltest.zip)
  └── gt/                       % semantic segmentation labels (gt_trainval.zip)
  ```
 
* **GTA5**: Download GTA5 images and ground truths from [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/). Please follow the dataset directory structure:
  ```html
  <GTA5_DIR>/                   % GTA5 dataset root
  ├── images/                   % input image 
  └── labels/                   % semantic segmentation labels
  ```
## Running IMEC 
IMEC can be run in three steps: pure source domain training, DKI training, and fine-tuning.
## source-only training
We directly use PODA’s pre-trained weights here. Please download the corresponding file from [PODA](https://github.com/astra-vision/PODA) and place it in the following location:
  ```html
  <IMEC_DIR>/                   % IMEC root
  ├── pretrain/                 % Place the weights there
  ```
  ## DKI training
  ```
sh DKI_training.sh
```
Please pay attention to modifying the data set path and training data set selection.

  ## ELP&SPC
  ```
sh ELP_SPC.sh
```

## Acknowledgment

We are very grateful to the [PODA](https://github.com/astra-vision/PODA), this project builds on their excellent foundation.


