# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward + backward) using [batch_size] number of samples
# e.g : 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch

"""
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms:
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
"""

import math

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

path = ''


class WineDataset(Dataset):

    # data loading
    def __init__(self, transform):
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])  # n_samples, 1
        self.n_samples = xy.shape[0]

        self.transform = transform

    # dataset[0]
    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # length of dataset
    def __len__(self):
        return len(self.n_samples)


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= inputs
        return inputs, targets


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


transform = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=transform)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i + 1) % 5 == 0:
            print(f"epoch {epoch + 1}/{epochs}, step {i + 1 / n_iterations}, inputs {inputs.shape}")



















