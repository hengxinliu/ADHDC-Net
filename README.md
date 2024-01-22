# Multiscale lightweight 3D segmentation algorithm with attention mechanism: Brain tumor image segmentation

This repository is the work of "_Multiscale lightweight 3D segmentation algorithm with attention mechanism: Brain tumor image segmentation_" based on **pytorch** implementation.The multimodal brain tumor dataset (BraTS 2019) could be acquired from [here](https://www.med.upenn.edu/cbica/brats-2019/).

## ADHDC-Net

<center>Architecture of  ADHDC-Net</center>
<div  align="center">  
 <img src="https://github.com/hengxinliu/ADHDC-Net/blob/main/fig/ADHDC-Net.jpg"
     align=center/>
</div>




## Requirements
* python 3.8
* pytorch 1.9.0
* nibabel
* pickle 
* imageio
* pyyaml

## Implementation

Download the BraTS2019 dataset and change the path:

```
experiments/PATH.yaml
```

### Data preprocess
Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance . 

```
python preprocess.py
```

(Optional) Split the training set into k-fold for the **cross-validation** experiment.

```
python split.py
```

### Training

Sync bacth normalization is used so that a proper batch size is important to obtain a decent performance. Multiply gpus training with batch_size=12 is recommended.The total training time is about 12 hours when using randomly cropped volumes of size 128×128×128 and on four Nvidia GTX2080Ti GPUs for 900 epochs.
d
```
python train_all.py --gpu=0,1,2,3 --cfg=ADHDC_Net --batch_size=12
```


### Test

You could obtain the resutls as paper reported by running the following code:

```
python test.py --mode=1 --is_out=True --verbose=True --use_TTA=True --postprocess=True --snapshot=True --restore=model_last.pth --cfg=ADHDC_Net --gpu=0
```
Then make a submission to the online evaluation server.



## Acknowledge

1. [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
2. [BraTS2018-tumor-segmentation](https://github.com/ieee820/BraTS2018-tumor-segmentation)
3. [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
4. [HDC-Net](https://github.com/luozhengrong/HDC-Net)

