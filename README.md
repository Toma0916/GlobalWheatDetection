# Global Wheat Detection

# Overview

This repository is for `Global Wheat Detection` that is Kaggle competition for detection of wheat segments in images from all over the world. (https://www.kaggle.com/c/global-wheat-detection)

<div align="center">
<img src="https://user-images.githubusercontent.com/18682053/82140037-b8573d80-9867-11ea-9acb-89bae6fb192e.png" title="OverView1">
</div>

<div align="center">
<img src="https://user-images.githubusercontent.com/18682053/82140038-b8efd400-9867-11ea-8618-b4abfb32640a.png" title="OverView2">
</div>


## Details

We implemented a pipeline for object detecion with pytorch. You can run training and inference easily in your machines.

In our pipeline, the following parameters are currently available, as well as TensorBoard and mlflow for visualization.

- Models
  -  Faster RCNN
  -  EfficientDet (based on https://github.com/rwightman/efficientdet-pytorch）
 
- Optimizers
  - Adam
  - SGD
  - AdamW

- Schedulers
  - StepLR
  - MultiStepLR
  - ExponentialLR
  - CosineAnnealingLR
  - WarmupMultiStepLR
  - WarmupCosineAnnealingLR
  - PiecewiseCyclicalLinearLR
  - PolyLR
  - WarmupCosineAnnealingRestartsLR
  
- Augmentations
  - HorizontalFlip (horizontal_flip)
  - VerticalFlip (vertical_flip)
  - Blur (blur)
  - MotionBlur (motion_blur)
  - MedianBlur (median_blur)
  - GaussianBlur (gaussian_blur)
  - OpticalDistortion (optical_distortion)
  - GridDistortion (grid_distortion)
  - ElasticTransform (elastic_transform)
  - CLAHE (clahe)
  - ChannelShuffle (channel_shuffle)
  - RandomGamma (random_gamma)
  - HueSaturationValue (hsv)
  - RGBShift (rgb_shift)
  - RandomContrast (random_contrast)
  - GaussNoise (gauss_noise)
  - Cutout (cutout)
  - Mosaic (mosaic)
  - CutMix (cutmix)
  - MixUp (mixup)

- Post processing
  - A filter based on confidence score
  - NMS (non maximum supression)
  - Soft-NMS
  - WBF (weighted boxes fusion)

- Other utils
  - TensorBoard (run `tensorboard --logdir ./output` in root)
  - MLFlow (run `mlflow ui --port 5000` in root)
 
## Requirement
- numba=0.49.1 
- numpy=1.18.1
- pandas=1.0.3
- opencv=4.2.0
- scikit-learn=0.22.1 
- scikit-optimize=0.7.4  
- pytorch=1.5.0 
- torchvision=0.6.0 
- albumentations=0.4.5
- mlflow=1.8.0 
- timm=0.1.26
- pycocotools=2.0.0
- omegaconf=2.0.0
- cython

Please see the requirements.txt. Run the following command to install.
pip install -U -r requirements.txt


## Usage
Specify your training configuration as a .json script, and then run the following command.

```
$ python train.py YOUR_CONFIG_PATH
```

You can use Single GPU.
Multi GPU will be available soon!

### Usage for EfficientDet
This model is from https://github.com/rwightman/efficientdet-pytorch.  
We used the following command to get the pretrained model (also from the above website)  
```
$ wget https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5-ef44aea8.pth
```
In order to use this, please specify the path to this weight in config.json. Example is given in ./sample_json/config_effdet.json.

