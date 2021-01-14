# Base (Reference) Framework for UG2+ Track 2.2 Challenge: Semi-Supervised Action Recognition in the Dark

This repository contains the framework for UG2+ Track 2.2 Challenge: Semi-supervised Action Recognition in the Dark.

## Prerequisites

This code is based on PyTorch, you may need to install the following packages:
```
PyTorch >= 1.2 (tested on 1.2/1.4/1.5/1.6)
opencv-python (pip install)
```

## Training

Training:
```
python train_arid_t2.py --network <Network Name>
```
- There are a number of parameters that can be further tuned. We recommend a batch size of 8 per GPU. Here we provide an example where the 3D-ResNet (18 layers) network is used and DANN is employed as the method for domain adaptation. The base network is directly ported from torchvision codes. You may use any other networks by putting the network into the /network folder. Do note that it is recommended you run the network once within the /network folder to debug before you run training.
- To employ other domain adaptation methods, you may need to change the code at train/model.py. You may refer to how DANN is employed.

## Testing

To generate the zipfile to be submitted, use the following commands:
```
cd predict
python predict_video.py
```
You may change the resulting zipfile name by changing the "--zip-file" configuration in the code, or simply by changing the configuration dynamically by
```
python predict_video.py --zip-file <YOUR PREFERRED ZIPFILE NAME>
```

## Other Information

- For more about the rules, regulations about this competition, do visit our site [here](http://cvpr2021.ug2challenge.org/track2.html)
- Our code base is adapted from [Multi-Fiber Network for Video Recognition](https://github.com/cypw/PyTorch-MFNet), we would like to thank the authors for providing the code base.
- You may contact me through cvpr2021.ug2challenge@gmail.com

