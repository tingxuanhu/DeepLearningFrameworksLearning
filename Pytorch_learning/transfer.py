import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

mean = np.array([.5, .5, .5])
std = np.array([.25, .25, .25])

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

data_dir = 'data/hymeoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

data_loader = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=8) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)


# model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = .0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # each epoch has a training a validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()

            running_loss = .0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss = loss.item() * inputs.size[0]
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]  # size varies between batch and batch
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc:{epoch_acc:.4f}")

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# ------------------------ option 1: fine tune -----------------
model = models.resnet18(pretrained=True)
# exchange the last fc layer
num_features = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_features, 2)  # override
model = model.to(device)  # move the model to GPU

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=.001)

step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25)


# ----- option 2: freeze, view the network as fixed feature extractor --------------
model_conv = models.resnet18(pretrained=True)
# freeze all the parameters in the beginning
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
# newly constructed modules have requires_grad = True by default
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=.001, momentum=.9)

# Decay LR by a factor of .1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)













