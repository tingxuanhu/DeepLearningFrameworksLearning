import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ------------------ TENSORBOARD ---------------------
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# ----------------------------------------------------


writer = SummaryWriter('runs/mnist1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

examples = iter(test_loader)
examples_data, examples_targets = next(examples)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(examples_data[i][0], cmap='gray')

# ------------------ TENSORBOARD ---------------------
img_grid = torchvision.utils.make_grid(examples_data)
writer.add_image('mnist_images', img_grid)
writer.close()

sys.exit()
# ----------------------------------------------------

