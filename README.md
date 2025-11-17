# AlexNet
Reimplementation of the 2012 AlexNet ([ImageNet Classification with Deep Convolutional Neural Networks][alexnet_paper]) paper by Hinton, Sutskever and Krizhevsky. 

[alexnet_paper]: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

<br>

## Architecture and deviations from the paper
True to the paper; eight layers of weights, the first five being convolutional layers, the ensuing three being fully connected. The third, fourth and fifth convolution layers are connected without any intervening pooling or normalization layers:
* Layer 1: 96 11x11x3 kernels (4px stride),
* Layer 2: 256 5x5x8 kernels,
* Layer 3: 384 3x3x256 kernels,
* Layer 4: 384 3x3x192 kernels,
* Layer 5: 256 3x3x192 kernels,
* Fully connected layers have 4096 neurons each.

<br>

There are some differences in my implementation worth noting, specifically:
* Using [CIFAR-10][dataset] instead of ImageNet for dataset. This also means only 10 image classes (as opposed to AlexNet's 1,000),
* To keep the same input dimensions from the paper, the 32x32px CIFAR-10 images were upscaled to 224x224 before feeding them into the network (this is done in `dataset.py` so no need for any manual activity),
* AlexNet uses a specific custom initialization (weights from a 0.01 std dev Gaussian and biases set to 0 or 1). Doing this meant my model failed to learn the dataset, so I'm using the PyTorch default Kaiming initialization,
* No GPU split: I simulated this using PyTorch groups, but the model was ultimately trained on my MBP M1 Max 32GB so no need to split,
* Hinton et al. manually monitored the validation error and "\[divided] the learning rate by 10 when the validation error rate stopped improving with the current learning rate": I am instead using a scheduler that automatically cuts the learning rate at epochs 5 and 8.

[dataset]: https://www.cs.toronto.edu/~kriz/cifar.html

<br>

## Results
TLDR, best validation accuracy of 76.56% with an average time of 760.33s (~12.7 mins) per epoch. Graphs and examples and such are stored in this repo's `results` folder. It's possible that upscaling the images has had a negative impact on expected accuracy (11x11 kernel looking for meaningful features in what may well be blurry interpolated slop). I have yet to verify this, but maybe will play around later for fun.

<br>

## Setup
Requirements for training and running the model are all in `requirements.txt` (you'll need to add matplotlib (and write some python) for generating plots if you want that).
1. Clone the repo, create your venv, install the dependencies,
2. My implementation uses a custom data loader that parses raw CIFAR-10 python binary files. You'll need to download and extract the data from `cifar-10-python.tar.gz`. You should get a folder titled `cifar-10-batches-py`, which you should store in a new folder named `data` in the project's root,
3. Run the training script (`python3 train.py`), wait a little while, then give it a try (`python3 predict.py my_image.jpg`).

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
