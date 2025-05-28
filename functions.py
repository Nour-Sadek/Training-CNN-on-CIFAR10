import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Find the per channel mean and standard deviation of a given data set that was
# fed to a data loader
def mean_std_per_channel(data_loader: DataLoader) -> tuple[torch.tensor, torch.tensor]:
    mean = 0
    std = 0
    for images, _ in data_loader:
        images_batch_size = images.size(0)
        images = images.view(images_batch_size, images.size(1), -1)
        mean = mean + images.mean(2).sum(0)  # sum over the batches (dim 0) so it is per channel
        std = std + images.std(2).sum(0)

    mean = mean / len(data_loader.dataset)
    std = std / len(data_loader.dataset)
    return mean, std


# Generate the appropriate transform depending on data set,
# as well as its mean and std
def apply_transforms(dataset, mean_std_tuple: tuple[torch.tensor, torch.tensor],
                     train: bool) -> transforms:
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std_tuple[0], std=mean_std_tuple[1])])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std_tuple[0], std=mean_std_tuple[1])])


# Apply He/Kaiming Initialization (Xavier Initialization with the extra /2)
# The default initialization is a Kaiming initialization using a uniform distribution,
# which should work with the RelU activation function as well
def initialize_using_norm_kaiming(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)


# Determine the accuracy of a model's predictions on a specific data set
def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    accuracy = round((correct / total) * 100, 2)
    return accuracy
