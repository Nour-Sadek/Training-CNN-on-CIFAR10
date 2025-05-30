import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from functions import mean_std_per_channel
from functions import apply_transforms
from functions import initialize_using_norm_kaiming
from functions import get_accuracy

from modified_models import CIFAR10Net_7

"""
Working on CIFAR10 dataset that is made up of images of 32x32 pixels in size
train set size = 45000
val set size = 5000
test set size = 10000
"""

# Hyperparameters
batch_size = 64
num_epochs = 100

# Variables
generator = torch.Generator().manual_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Preparing and pre-processing the data ny normalizing the pixel values per channel and applying On-the-fly 
augmentation on the training images"""
# Finding out the mean and standard deviation of the CIFAR10 dataset so that I would normalize based on that
transform = transforms.ToTensor()

# Get the train and test data sets
trainset = torchvision.datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())

# Divide the train set further into train set and validation set
train_size = int(0.9 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)

# Load the three datasets separately
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Calculate the mean and standard deviation for each dataset
train_mean_std = mean_std_per_channel(train_loader)
val_mean_std = mean_std_per_channel(val_loader)
test_mean_std = mean_std_per_channel(test_loader)

# Transform the datasets again with normalization,
# as well as On-the-fly/Online augmentation
trainset.dataset.transform = apply_transforms(train_mean_std, train=True)
valset.dataset.transform = apply_transforms(val_mean_std, train=False)
testset.transform = apply_transforms(test_mean_std, train=False)

# Load the datat sets again
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Sanity checks to make sure that the data for the three datasets was normalized correctly. We expect the means to be
# close to 0 and the standard deviations to be close to 1
print(mean_std_per_channel(test_loader))
print(mean_std_per_channel(val_loader))
print(mean_std_per_channel(train_loader))

"""After data has been pre-processed and is ready, it is time to create the model"""


class CIFAR10Net(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 40, 4)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.fc1 = nn.Linear(40 * 6 * 6, 124)
        self.bnfc1 = nn.BatchNorm1d(124)
        self.fc2 = nn.Linear(124, 64)
        self.bnfc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Run through CNN
        x = self.bn1(self.conv1(x))
        x = self.pool(self.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(self.relu(x))

        # Run through FC network
        x = torch.flatten(x, 1)
        x = self.relu(self.bnfc1(self.fc1(x)))
        x = self.relu(self.bnfc2(self.fc2(x)))
        x = self.fc3(x)

        return x


# Initializing the CIFAR10 model and applying Kaiming initialization using normal distribution
model = CIFAR10Net()
model.to(device)
model.apply(initialize_using_norm_kaiming)

# Determining the loss and optimization functions to be used during training
criterion = nn.CrossEntropyLoss()
# Values of lr and betas are already the default values of the hyperparameters, but I have them explicitly stated here
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)

"""
# Training the model
for epoch in range(num_epochs):
    running_loss = 0
    for i, data in enumerate(train_loader, 1):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()
        if i % 200 == 0:
            print(f'[{epoch + 1}, {i:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0

print("Finished Training")
"""

# If I want to compute the loss and accuracy per epoch of the train and val sets during training
train_losses_epoch = []
val_losses_epoch = []

train_accuracy_epoch = []
val_accuracy_epoch = []

for epoch in range(num_epochs):
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
        optimizer.step()

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
    val_accuracy_epoch.append(get_accuracy(model, val_loader))

    # Print the statistics after one epoch
    print(
        f"After epoch {epoch + 1}: training loss = {train_losses_epoch[epoch]}, validation loss = {val_losses_epoch[epoch]}, training accuracy = {train_accuracy_epoch[epoch]}, validation accuracy = {val_accuracy_epoch[epoch]}")

print("Finished Training")

"""Plot the train and validation loss and accuracy"""
epochs = range(1, len(train_losses_epoch) + 1)

# Plot Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses_epoch, label='Train Loss')
plt.plot(epochs, val_losses_epoch, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy_epoch, label='Train Accuracy')
plt.plot(epochs, val_accuracy_epoch, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
path = "./cifar_net.pth"
torch.save(model.state_dict(), path)

# If I wanted to load back the model
# loaded_model = CIFAR10Net()
# loaded_model.load_state_dict(torch.load(path, weights_only=True))


"""Dimensionality-Reduction Analysis"""

# Load the 7th model that didn't overfit the training data set as much as the first version of the model
loaded_model = CIFAR10Net_7()
loaded_model = loaded_model.to(device)
loaded_model.load_state_dict(torch.load("CIFAR_7.pth", weights_only=True))

# Get the features of the second to last fully-connected layer
loaded_model.eval()
features = []
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, fc2_features = loaded_model(inputs, return_features=True)
        _, predicted = torch.max(outputs, 1)
        features.append(fc2_features)
        predictions.append(predicted)
        true_labels.append(labels)

# Concatenate the tensors of the batches as one tensor
predictions = torch.cat(predictions, dim=0)
features = torch.cat(features, dim=0)
true_labels = torch.cat(true_labels, dim=0)
is_correct = true_labels == predictions

# Draw the t-SNE plot
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features.cpu().numpy())

tsne_df = pd.DataFrame({
    "x": features_2d[:, 0],
    "y": features_2d[:, 1],
    "predictions": predictions.cpu().numpy(),
    "label": true_labels.cpu().numpy(),
    "correct": is_correct.cpu().numpy()
})

# Define a map linking numeric labels to class names
class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
label_map = {i: name for i, name in enumerate(class_names)}

# Replace numeric predictions with class names in the DataFrame for plotting
tsne_df["predictions"] = tsne_df["predictions"].map(label_map)

plt.figure(figsize=(10, 8))

# Plot correct predictions (no border)
sns.scatterplot(
    data=tsne_df[tsne_df["correct"]], x="x", y="y",
    s=10, hue="predictions", palette="tab10", edgecolor=None, linewidth=0,
    legend="full"
)

# Plot incorrect predictions (with black border)
sns.scatterplot(
    data=tsne_df[~tsne_df["correct"]], x="x", y="y",
    s=10, hue="predictions", palette="tab10", edgecolor="black", linewidth=0.5,
    legend=False
)

plt.title("t-SNE of fc2 Features with Misclassification Highlighted")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("t-SNE on CIFAR_7 model.png", dpi=300, bbox_inches="tight")
