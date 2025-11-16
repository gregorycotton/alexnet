import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle
import os
import numpy as np

class Cifar10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform_stats=None):
        self.data_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        self.train = train
        
        if transform_stats is None:
            self.normalize_mean = torch.tensor([0.485, 0.456, 0.406])
            self.normalize_std = torch.tensor([0.229, 0.224, 0.225])
        else:
            self.normalize_mean = transform_stats['mean']
            self.normalize_std = transform_stats['std']

        self.data = []
        self.labels = []

        if self.train:
            # Load my 5 training batches
            for i in range(1, 6):
                batch_file = os.path.join(self.data_dir, f'data_batch_{i}')
                self._load_batch(batch_file)
        else:
            # Load my test batch
            batch_file = os.path.join(self.data_dir, 'test_batch')
            self._load_batch(batch_file)

        # Convert to numpy arrays
        self.data = np.vstack(self.data)
        self.labels = np.hstack(self.labels)

        self.data = self.data.reshape(-1, 3, 32, 32)

    def _load_batch(self, filepath):
        with open(filepath, 'rb') as f:
            # Unpickle aw yea
            batch = pickle.load(f, encoding='bytes')
            
            self.data.append(batch[b'data'].copy())
            self.labels.append(batch[b'labels'].copy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # We get the numpy array and convert it to a torch tensor
        image = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Normalize pixel values
        image = image / 255.0

        if self.train:
            if torch.rand(1) > 0.5:
                image = torch.flip(image, dims=[2])

        # Resize image from 32x32 to 224x224
        image = F.interpolate(image.unsqueeze(0), 
                              size=(224, 224), 
                              mode='bilinear', 
                              align_corners=False).squeeze(0)
        
        # Reshape mean/std to (3, 1, 1) to broadcast over the (3, 224, 224) image
        mean = self.normalize_mean.view(3, 1, 1)
        std = self.normalize_std.view(3, 1, 1)
        image = (image - mean) / std
        
        return image, label