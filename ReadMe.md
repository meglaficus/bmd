# Bone Metastasis Detector
This repository provide code for train and test deep learning-based algorithm for bone metastasis detection on whole-body CT.

## Environment
This package was created and tested on the following environment.
* Ubuntu 18.04.3 LTS
* Nvidia driver 450.102.04
* CUDA  10.0
* cuDNN 7.6.5
* Python 3.6.5
* tensorflow-gpu 1.13.1
* Keras 2.2.5

## Dataset Structure

To train 3D UNet, dataset is required to organize in the following structure.  Numpy files consist of 96 x 96 x 96 voxel CT images including bone metastasis lesions, and corresponding lesion labels. 
```
dir1
├─ train
│ ├─ scan_001
│ │ ├─ image_001_ct.npy
│ │ ├─ image_001_lb.npy
│ │ ├─ image_002_ct.npy
│ │ ├─ image_002_lb.npy
│ │ ├─ image_003_ct.npy
│ │ ├─ image_003_lb.npy
│ │ │  ...
│ │ ├─ image_xxx_ct.npy
│ │ └─ image_xxx_lb.npy
│ ├─ scan_002
│ ├─ scan_003
│ │  ...
│ └─ scan_xxx
└─ val
  ├─ scan_001
  ├─ scan_002
  ├─ scan_003
  │  ...
  └─ scan_xxx
```
<br>

To train 3D ResNet, dataset is required to organize in the following structure.  Numpy files consist of 32 x 32 x 32 voxel CT images with or without bone metastasis lesion. 
```
dir2
├─ train_pos
│ ├─ scan_001
│ │ ├─ image_001.npy
│ │ ├─ image_002.npy
│ │ ├─ image_003.npy
│ │ │  ...
│ │ └─ image_xxx.npy
│ ├─ scan_002
│ ├─ scan_003
│ │  ...
│ └─ scan_xxx
├─ train_neg
│ ├─ scan_001
│ ├─ scan_002
│ │  ...
│ └─ scan_xxx
├─ val_pos
└─ val_neg
```
<br>

Test images are required to organize in the following structure. Numpy files consist of 3D CT images of each scan, with corresponding bone segmentation label and bone metastasis region labels (ground truth labels). All images are assumed to have been converted into 1 mm iso voxel images before testing.
```
dir3
├─ scan_001_ct.npy
├─ scan_001_bs.npy
├─ scan_001_lb.npy
├─ scan_002_ct.npy
├─ scan_002_bs.npy
├─ scan_002_lb.npy
│  ...
├─ scan_xxx_ct.npy
├─ scan_xxx_bs.npy
└─ scan_xxx_lb.npy
```

## Train
To train models, run "train_**.py" with specifying the directory paths of training data as arguments. For example: 
```
python train_3d_unet.py --root_dir /xxx/dir2/ --train_dir train/ --val_dir val/
```

## Test
To test models, run "test_unet_and_resnet.py" with specifying the paths of test data and trained models as arguments. The threshold for the predicted probability of 3D ResNet can be specified with the argument of "thr".
```
python test_unet_and_resnet.py --test_images_dir /xxx/dir3/ --unet_model /yyy/unet3d_model.h5 --resnet_model /zzz/resnet3d_model_1.h5 /zzz/resnet3d_model_2.h5 /zzz/resnet3d_model_3.h5 --thr 0.6
```

## Acknowledgements
The code for the RICAP function was quoted from the repository of [koshian2](https://github.com/koshian2/keras-ricap), with some edits.

## License
Sorry, there is no license offered for this project. This means I do not give you permission to copy or reuse the code in this project. This does not apply when you aim to retest the experiment of the paper. Commercial use is strictly prohibited.

[https://choosealicense.com/no-permission/](https://choosealicense.com/no-permission/)
