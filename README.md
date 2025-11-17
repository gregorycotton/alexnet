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

## Results
TLDR, best validation accuracy of 76.56% with an average time of 760.33s (~12.7 mins) per epoch. Graphs and examples and such are stored in this repo's `results` folder.

## Setup
Requirements for training and running the model are all in `requirements.txt` (you'll need to add matplotlib for generating plots).
1. Clone the repo, create your venv, install the dependencies,
2. My implementation uses a custom data loader that parses raw CIFAR-10 python binary files. You'll need to download and extract the data from `cifar-10-python.tar.gz`. You should get a folder titled `cifar-10-batches-py`, which you should store in a new folder named `data` in the project's root,
3. Run the training script (have made it hardware agnostic), wait a little while, then give it a try.

When all is said and done your project structure should look something like the below.

```text
alexnet/
├── .gitignore
├── README.md
├── requirements.txt
├── venv/                        
├── data/
│   └── cifar-10-batches-py/     
│       ├── batches.meta
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       └── test_batch
├── dataset.py                   
├── model.py                     
├── train.py                     
├── predict.py                  
└── alexnet_cifar10.pth          