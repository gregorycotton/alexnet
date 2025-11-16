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


## Architecture & deviations from the paper
True to the paper; eight layers of weights, the first five being convolutional layers, the ensuing three being fully connected. The third, fourth and fifth convolution layers are connected without any intervening pooling or normalization layers:
* Layer 1: 96 11x11x3 kernels (4px stride),
* Layer 2: 256 5x5x8 kernels,
* Layer 3: 384 3x3x256 kernels,
* Layer 4: 384 3x3x192 kernels,
* Layer 5: 256 3x3x192 kernels,
* Fully connected layers have 4096 neurons each.

There are some differences in my implementation worth noting, specifically:
* As mentioned above, using CIFAR-10 instead of ImageNet for dataset. This also means instead of 1,000 image classes I use 10,
* To keep the same input dimensions from the paper, the 32x32px CIFAR-10 images were upscales to 224x224 before feeding them into the network,
* Hinton and company use a specific custom initialization (weights from a 0.01 std dev Gaussian and biases set to 0 or 1). Doing this meant my model failed to learn the dataset, so I'm using the PyTorch default Kaiming initialization,
* No GPU split: I simulated this using PyTorch groups, but the model was ultimately trained on my MBP M1 Max,
* Hinton et al. manually monitored the validation error and "\[divided] the learning rate by 10 when the validation error rate stopped improving with the current learning rate": I am instead using a scheduler that automatically cuts the learning rate at epochs 5 and 8.

Otherwise I tried to basically do whatever was done in the paper idk man you can read it.

## Results & additional considerations
As I am up-scaling images from 32x32 to 224x224, there is a negative impact on expected accuracy (11x11 kernel looking for meaningful features is looking at blurry interpolated slop). To try and quantify the degree to which I've traded faithfulness to the paper for performance, I retrained the model with no image up-scaling and an more appropriately sized kernel for the 32x32px images (pooling layers adjusted as well: all the changes made for this purpose are in the commented in the model.py file). Quick comparison below between the faithful versus optimized model. 

| Model        | Best Validation Accuracy | Avg. Time per Epoch |
|--------------|--------------------------|---------------------|
| Old faithful | 76.56%                   | ~760 seconds        |
| Optimized    | TBD                      | TBD                 |

Note: the code in this repo is the faithful model.

## Setup
text