# Global Wheat Detection

# Overview

This repository is for `Global Wheat Detection` that is Kaggle competition for detection of wheat segments in images. (https://www.kaggle.com/c/global-wheat-detection)

<div align="center">
<img src="https://user-images.githubusercontent.com/18682053/82140037-b8573d80-9867-11ea-9acb-89bae6fb192e.png" title="OverView1">
</div>

<div align="center">
<img src="https://user-images.githubusercontent.com/18682053/82140038-b8efd400-9867-11ea-8618-b4abfb32640a.png" title="OverView2">
</div>


## Details

We implemented pipeline for object detecion by using pytorch. You can run training and inference very easily in your machines.

In our pipeline, you can use bellow with friendly setting scheme and check result with TensorBoard.

- model
  -  Faster RCNN
  -  SGD
  -
 
- optimizer
  - Adam
  -
  - 

- scheduler
  - StepLR
  - MultiStepLR
  - ExponentialLR
  - CosineAnnealingLR
  - WarmupMultiStepLR
  - WarmupCosineAnnealingLR
  - PiecewiseCyclicalLinearLR
  - PolyLR
  
- augmentations
  - HorizontalFlip
  - VerticalFlip
  - Blur
  - MotionBlur
  - MedianBlur
  - GaussianBlur
  - OpticalDistortion
  - GridDistortion
  - ElasticTransform
  - CLAHE
  - ChannelShuffle
  - RandomGamma
  - HueSaturationValue
  - RGBShift
  - RandomContrast
  - GaussNoise
  - Cutout
  - Mosaic
 
 
## Requirement
- numba=0.49.1 
- numpy=1.18.1
- pandas=1.0.3
- opencv=4.2.0
- scikit-learn=0.22.1 
- pytorch=1.5.0 
- torchvision=0.6.0 
- albumentations=0.4.5
- mlflow=1.8.0 

## Usage
Make json file about train setting and run hogehoge.sh with json path.

You can use Single GPU.
Multi GPU will be available soon!


