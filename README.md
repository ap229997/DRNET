## DRNET
PyTorch implementation of the NIPS 2017 paper "Unsupervised Learning of Disentangled Representations from Video" 
[[Link]](https://papers.nips.cc/paper/7028-unsupervised-learning-of-disentangled-representations-from-video.pdf) </br>

### Introduction
The authors present a new model DRNET that learns disentangled image representations from video. They utilise a novel adversarial loss to learn a representation that factorizes each frame into a stationary part and a temporally varying component. They evaluate the approach on a range of synthetic (MNIST, SUNCG) and real (KTH Actions) videos, demonstrating the ability to coherently generate hundreds of steps into the future. </br>

### Setup
This repository is compatible with python 2. </br>
- Follow instructions outlined on [PyTorch Homepage](https://pytorch.org/) for installing PyTorch (Python2). 
- Follow instructions outlined on [Lua Download page](https://www.lua.org/download.html) for installing Lua. This is required since the script for converting the KTH Actions dataset is provided in Lua. </br>

### Downloading and Preprocessing data
Detailed instructions for downloading and preprocessing data are provided by [edenton/drnet](https://github.com/edenton/drnet).

##### KTH Actions dataset
Download the KTH action recognition dataset by running:
```
sh datasets/download_kth.sh /my/kth/data/path/
```
where /my/kth/data/path/ is the directory the data will be downloaded into. Next, convert the downloaded .avi files into .png's for the data loader. To do this you'll want ffmpeg installed. Then run:
```
th datasets/convert_kth.lua --dataRoot /my/kth/data/path/ --imageSize 128
```
The ```--imageSize``` flag specifiec the image resolution. The models implemented in this repository are for image size 128 or greater. However, they can also be used for lesser image resolution by decreasing the number of convolution blocks in the network architecture.

The file ```utils.py``` contains the dataloader for processing KTH Actions data further. 

##### MNIST, SUNCG
The file ```utils.py``` contains the functionality for downloading and processing the MNIST and SUNCG datasets while running the model. </br>

### Train the model
Different architectures are used for training the model on different datasets. This can be set by specifying the ```--dataset``` parameter while calling ```main.py```. Different networks are used in the paper - base, base with skip connections, lstm for sequential predictions. These models can be trained by running the following commands: (Other parameters to be specified are described in ```main.py```. Refer to it for better understanding) </br>
- Training the base model ```python main.py```
- Training the base model with skip connections ```python main.py --use_skip True```
- Training the lstm model for sequential predictions ```python main.py --use_lstm True``` </br> 

### Training loss curves
Training loss curves for (*left*) reconstruction loss, (*center*) similarity loss (with the base model) and (*right*) mse loss (with lstm model) vs the number of iterations are shown here. These results are on the MNIST dataset.
<p align="center">
  <img src="https://github.com/ap229997/DRNET/blob/master/saved_results/rec_loss.png" width="250"/>
  <img src="https://github.com/ap229997/DRNET/blob/master/saved_results/sim_loss.png" width="250"/>
  <img src="https://github.com/ap229997/DRNET/blob/master/saved_results/lstm_loss.png" width="250"/>
</p>
Also, on the running the training code for MNIST provided by [edenton/drnet-py] for 10 epochs, the corresponding loss values obtained are shown here. Each epoch constitutes 600 iterations. 
<p align="center">
  <img src="https://github.com/ap229997/DRNET/blob/master/saved_results/original_loss.png" width="400">
</p>
The values for the similarity loss goes quickly to zero in both the cases and good convergence can be observed in case of reconstruction loss and mse loss for the lstm model.

### Alternate Implementations
Alternate implementation for this paper are also available. Refer to them for better understanding.
- Lua version - https://github.com/edenton/drnet
- PyTorch version - https://github.com/edenton/drnet-py </br>

### Citation
If you find this code useful, please consider citing the original work by authors:
```
@incollection{Denton2017NeurIPS,
title = {Unsupervised Learning of Disentangled Representations from Video},
author = {Denton, Emily L and Birodkar, vighnesh},
booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
year = {2017}
}
```
