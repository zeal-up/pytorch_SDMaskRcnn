# pytorch_SDMaskRcnn

This repository is a pytorch version of sd-maskrcnn and maskrcnn.
The offical sd-maskrcnn code is written by tensorflow.(refer to https://github.com/BerkeleyAutomation/sd-maskrcnn).
And the maskrcnn backbone is mostly adapted from https://github.com/darolt/mask_rcnn. I make some modification as below:
1. Reorganized the code struture to more pytorch style.
2. I replace some C/CUDA function with pytorch in-build functions(NMS && RoiAlign)

The code can run with multi-gpus and I tested on 
+ pytorch 1.4.0
+ torchvision 0.5.0
+ python 3.6

## Training
+ download dataset(wisdom-sim) from sd-maskrcnn repo(https://github.com/BerkeleyAutomation/sd-maskrcnn)
+ make some config change in `./datasets/wisdom/wisdomConfig.yml`(change the dataset path)
+ run with `CUDA_VISIBLE_DEVICES=X,X python main/trian.py`
+ all training log will store in `./log` directory.
+ install `tensorboardX` package and you can monitor the training procedure in browser.

## Inference
+ download the inference data(wisdom-real) from sd-maskrcnn repo.
+ change the dataset path in `./datasets/wisdom/widomInference.yml`
+ run with `CUDA_VISIBLE_DEVICES=0 python main/inference.py`