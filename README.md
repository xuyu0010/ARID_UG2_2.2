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
__Updates for Jul 2022__
- The UG2+ 2022 challenge has ended. Congratulations on all winning teams and thank you for your support. Due to limited resources and other technical factors, ARID will NOT be holding any new competitions in the near future. We hope to update/upgrade ARID soon for future challenges as soon as possible! In the meantime, feel free to recap this year's workshop and challenge [here](http://cvpr2022.ug2challenge.org)!
- We will like to update you with the latest statistics of ARID (v1.5) as follows:
```
mean = [0.05131350665077962, 0.04543643187320746, 0.043676298767677715]
standard deviation = [0.07904116281533713, 0.07485863367941209, 0.06953901191852299]
```
- The original ARID statistics are listed for comparison:
```
mean = [0.079612, 0.073888, 0.072454]
standard deviation = [0.100459, 0.09705, 0.089911]
```

__Updates for Jan 2022__
- For more about the rules, regulations about this competition, do visit our site [here](http://cvpr2022.ug2challenge.org/track2.html)
- Our code base is adapted from [Multi-Fiber Network for Video Recognition](https://github.com/cypw/PyTorch-MFNet), we would like to thank the authors for providing the code base.
<!-- - You may contact me through cvpr2022.ug2challenge@gmail.com -->

