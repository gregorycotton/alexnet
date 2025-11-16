# AlexNet
Reimplementation of the 2012 "[ImageNet Classification with Deep Convolutional Neural Networks][alexnet_paper]" paper by Hinton, Sutskever and Krizhevsky. 

[alexnet_paper]: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf


## Dataset
I used the [CIFAR-10 dataset][dataset] for training images.
* 60000 32x32 colour images in 10 classes
* 6000 images per class
* 50000 training images
* 10000 test images

[dataset]: https://www.cs.toronto.edu/~kriz/cifar.html


## Architecture
As true to the paper as I could. This means eight layers of weights, the first five being convolutional layers, the ensuing three being fully connected. The third, fourth and fifth convolution layers are connected without any intervening pooling or normalization layers.
* Layer 1: Filters 224x224x3 input image with 96 11x11x3 kernels, with a 4px stride,
* Layer 2: 256 5x5x8 kernels,
* Layer 3: 384 3x3x256 kernels,
* Layer 4: 384 3x3x192 kernels,
* Layer 5: 256 3x3x192 kernels,
* Fully connected layers have 4096 neurons each.

It's worth noting that for AlexNet, the kernels of the second, fourth, and fifth convolutional layers are connected only to those kernel maps in the previous layer which reside on the same GPU. The kernels of the third convolutional layer are connected to all kernel maps in the second layer. I'm training on my MacBook Pro (M1 Max, 32GB) and have no need to split GPUs but to try and be faithful, I will simulate this split using groups in PyTorch.

This otherwise means basically whatever else was done in the paper idk man you can read it.

## Setup
text
