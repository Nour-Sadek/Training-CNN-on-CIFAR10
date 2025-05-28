import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

import torch.nn as nn
import torchvision.models as models
import numpy as np

from functions import mean_std_per_channel
from functions import get_accuracy

# Variables
generator = torch.Generator().manual_seed(20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Use Transfer Learning where the Convolutional Network is treated as a fixed feature extractor and only the last 
fully-connected layer (that links to the classifications) is re-trained to fit CIFAR-10's 10 classification options."""
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)


# Generate the appropriate transform depending on data set, as well as its mean and std
# ResNet18 expects an input of 224x224 rather than 32x32
def apply_transforms(dataset, mean_std_tuple: tuple[torch.tensor, torch.tensor],
                     train: bool) -> transforms:
    if train:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std_tuple[0], std=mean_std_tuple[1])])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std_tuple[0], std=mean_std_tuple[1])])


# Finding out the mean and standard deviation of the CIFAR10 dataset so that I
# would normalize based on that
transform = transforms.ToTensor()

# Get the train and test datatsets
trainset = torchvision.datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())

# Divide the trainset further into trainset and validation set
train_size = int(0.9 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)

# Load the three datasets separately
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = DataLoader(valset, batch_size=64, shuffle=False)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Calculate the mean and standard deviation for each dataset
train_mean_std = mean_std_per_channel(train_loader)
val_mean_std = mean_std_per_channel(val_loader)
test_mean_std = mean_std_per_channel(test_loader)

# Transform the datasets again with normalization,
# as well as On-the-fly/Online augmentation
trainset.dataset.transform = apply_transforms(trainset, train_mean_std, train=True)
valset.dataset.transform = apply_transforms(valset, val_mean_std, train=False)
testset.transform = apply_transforms(testset, test_mean_std, train=False)

# Load the datatsets again
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = DataLoader(valset, batch_size=64, shuffle=False)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Train the last fully-connected layer of the resnet model
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)

train_losses_epoch = []
val_losses_epoch = []

train_accuracy_epoch = []
val_accuracy_epoch = []

val_accuracy = get_accuracy(model, val_loader)

for epoch in range(100):
    # Training for one epoch and determining the loss of training set over this epoch
    train_losses_step = []
    model.train()
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(val_accuracy)

        train_losses_step.append(loss.item())
    train_losses_epoch.append(np.array(train_losses_step).mean())

    # Determining the loss of validation set over this epoch
    model.eval()
    val_losses_step = []
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses_step.append(loss.item())
        val_losses_epoch.append(np.array(val_losses_step).mean())

    # Determining the accuracy of the train and val datasets on the model after training for this epoch
    train_accuracy_epoch.append(get_accuracy(model, train_loader))
    val_accuracy = get_accuracy(model, val_loader)
    val_accuracy_epoch.append(val_accuracy)

    # Print the statistics after one epoch
    print(
        f"After epoch {epoch + 1}: training loss = {train_losses_epoch[epoch]}, validation loss = {val_losses_epoch[epoch]}, training accuracy = {train_accuracy_epoch[epoch]}, validation accuracy = {val_accuracy_epoch[epoch]}")

print("Finished Training")

# Find the accuracy of using Transfer Learning on the test set
print(f"Accuracy of the network on the test images: {get_accuracy(model, test_loader)} %")
