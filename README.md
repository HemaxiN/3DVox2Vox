# 3DVox2Vox
Synthetic Generation of 3D Microscopy Images using Generative Adversarial Networks

This repository contains the Python implementation of a Vox2Vox model to generate 3D microscopy images containing nuclei and Golgi.

## Architecture

We use a Vox2Vox model to generate 3D microscopy images of nuclei and Golgi. The generator (G) is a 3D U-Net model. It receives an input image (x) that contains the conditions to generate the synthetic image G(x). The discriminator (D) is a PatchGAN model. It receives pairs {x, G(x)} or {x,y} and classifies as real or synthetic translation of x. 

![](https://github.com/HemaxiN/3DVox2Vox/blob/main/images/architecture.png)

## Input Images

Im this work we compared four approaches to generate synthetic microscopy images of nuclei and Golgi.  All the approaches are based on the Vox2Vox model, the only difference between them is the input image (x) that contains the conditions to generate the synthetic images. The four input images considered in this work are the following:

![](https://github.com/HemaxiN/3DVox2Vox/blob/main/images/input_images.png)


## Requirements

Python 3.5.2, Keras 2.2.4 and other packages listed in `requirements.txt`.

## Training on your own dataset

Change the parameters `train_dir` and `val_dir` in the file `train_main.py` amd run it, where `train_dir` and `val_dir` are organized as follows:

```
train_val_dataset
├── train
│   ├── images
│   └── masks
└── val
    ├── images
    └── masks
```
where images are .tif files and correspond to microscopy image patches of size (64,256,256,3), and masks are .tif files corresponding to input image patches of size (64,256,256,3).

## Evaluation

To generate images using a trained G model, change the following parameters in the file `test_main.py`and run it:

* `models_path`: path to the trained G model; 
* `input_path`: directory containing input images for G;
* `save_dir`: directory where the synthetic images will be saved.

To compute the Fréchet Inception Distance (FID) score, change the parameters `img1_dir` and `img2_dir` in the file `fid_score_main.py`, where:

* `img1_dir`: contain the real images (64,256,256,3);
* `img2_dir`: contain the corresponding synthetic images generated by the trained G model (64, 256, 256, 3).

and run the file `fid_score_main.py`.
