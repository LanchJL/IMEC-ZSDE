# IMEC-ZSDE

## News
*  2025.3.11 The project is not yet fully uploaded. We are actively organizing the code and working towards making it completely open source.

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

## Installation
Initially, please make sure your environment containing the required packages.	
If not, you can run:
```
$ cd repository
$ pip install -r requirements.txt
```

Then activate environment using:
```
$ conda activate env_name
```

## Running IMEC 
The IMEC algorithm consists of three steps: source-only training, DKI training, and fine-tuning.	In the actual implementation, we provide detailed information for each step of separation to facilitate debugging.	

### source-only training
We directly use PODA's pre-trained weights here. Please download the corresponding file from [PODA](https://github.com/astra-vision/PODA) and place it in the following location:
  ```html
  <IMEC_DIR>/                   % IMEC root
  ├── pretrain/                 % Place the weights there
  ```
  ### DKI training
  ```
$ cd run/
$ sh DKI_training.sh
```
Please pay attention to several main settings here: 
* dataset        %the source dataset for training (cityscapes or gta5)
* data_root        %root to your dataset
* DKI_save_dir        %Dir used to store DKI weights
* target_domain        %The domain IMEC wants to extend

All prompt words are stored in Prompts/, and you can replace them with your own prompt words by modifying the corresponding files.

  ### ELP&SPC
  ```
$ cd run/
$ sh ELP_SPC.sh
```
Please note the following settings for this step:
* dataset        %the source dataset for training (cityscapes or gta5)
* data_root        %root to your dataset
* DKI_checkpoints        %the path to DKI checkpoints
* Style_dir        %Dir used to store styles


## Acknowledgment
We are grateful to the [PODA](https://github.com/astra-vision/PODA) for their excellent foundation, which has been instrumental in the development of our project.

