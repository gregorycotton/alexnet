import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            # 96 kernels, 11x11, stride 4. Input 224x224x3
            # AlexNet takes input of 224x224, but first convolution (11x11, stride 4)
            # only works if the output is 55x55 so we will set padding to 2
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # LRN after 1st layer
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            # Max-pooling after LRN
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 2
            # 256 kernels, 5x5, connected only on same GPU
            # Groups=2. Input is 96 (48 per group)
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            # LRN after 2nd layer
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            # Max-pooling after LRN
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 3
            # 384 kernels, 3x3, connected to all
            # Groups=1
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4
            # 384 kernels, 3x3, connected only on same GPU
            # Groups=2.
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            # Layer 5
            # 256 kernels, 3x3, "connected only on same GPU"
            # Groups=2
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            # Max-pooling after 5th conv layer
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            # Dropout in first two fully connected layers
            nn.Dropout(p=0.5),
            # 4096 neurons
            # 256 * 6 * 6 input
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
            # The 1000-way softmax is inside nn.CrossEntropyLoss
        )
        
        # This is fuckin my shit up rn
        # self.apply(self._init_weights)

    def forward(self, x):
        x = self.features(x)
        
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)
        return x

    # get rid of this too
    # def _init_weights(self, module):
    #     """
    #     Initializes weights and biases as described in the AlexNet paper.
    #     Weights: zero-mean Gaussian with std 0.01
    #     Biases: 0 for Conv 1, 3.
    #             1 for Conv 2, 4, 5 and all FC layers.
    #     """
        
    #     if isinstance(module, nn.Conv2d):
    #         # Initialize weights
    #         nn.init.normal_(module.weight, mean=0.0, std=0.01)
            
    #         if module.bias is not None:
    #             # Check if groups=2
    #             if module.groups == 2:
    #                 nn.init.constant_(module.bias, 1.0)
    #             else:
    #                 nn.init.constant_(module.bias, 0.0)

    #     # Check if fully connected
    #     elif isinstance(module, nn.Linear):
    #         nn.init.normal_(module.weight, mean=0.0, std=0.01)
            
    #         if module.bias is not None:
    #             nn.init.constant_(module.bias, 1.0)