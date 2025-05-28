import torch
import torch.nn as nn


class CIFAR10Net_2(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 40, 4)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.fc1 = nn.Linear(40 * 6 * 6, 32)
        self.bnfc1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bnfc2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 10)

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


class CIFAR10Net_3(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 40, 4)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.fc1 = nn.Linear(40 * 6 * 6, 10)

        # Activation function*
        self.relu = nn.ReLU()

    def forward(self, x):
        # Run through CNN
        x = self.bn1(self.conv1(x))
        x = self.pool(self.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(self.relu(x))
        # Run through FC network
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class CIFAR10Net_4(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        # FC layers
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bnfc1 = nn.BatchNorm1d(64)
        self.bnfc2 = nn.BatchNorm1d(32)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Run through CNN
        x = self.bn1(self.conv1(x))
        x = self.bn1(self.conv2(self.relu(x)))
        x = self.pool(self.relu(x))
        x = self.bn2(self.conv3(x))
        x = self.bn2(self.conv4(self.relu(x)))
        x = self.pool(self.relu(x))

        # Run through FC network
        x = torch.flatten(x, 1)
        x = self.relu(self.bnfc1(self.fc1(x)))
        x = self.relu(self.bnfc2(self.fc2(x)))
        x = self.fc3(x)
        return x


class CIFAR10Net_5(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        # FC layers
        self.fc1 = nn.Linear(64 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bnfc1 = nn.BatchNorm1d(32)
        self.bnfc2 = nn.BatchNorm1d(16)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Run through CNN
        x = self.bn1(self.conv1(x))
        x = self.bn1(self.conv2(self.relu(x)))
        x = self.pool(self.relu(x))
        x = self.bn2(self.conv3(x))
        x = self.bn2(self.conv4(self.relu(x)))
        x = self.pool(self.relu(x))

        # Run through FC network
        x = torch.flatten(x, 1)
        x = self.relu(self.bnfc1(self.fc1(x)))
        x = self.fc2(x)
        return x


class CIFAR10Net_6(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        # FC layers
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bnfc1 = nn.BatchNorm1d(64)
        self.bnfc2 = nn.BatchNorm1d(32)

        # Activation function
        self.relu = nn.ReLU()

        # Residual Block Linear Projection
        self.res_proj1 = nn.Conv2d(3, 16, 9)
        self.res_proj2 = nn.Conv2d(16, 32, 5)

    def forward(self, x):
        # Run through CNN
        # Residual Block 1
        input = x
        x = self.bn1(self.conv1(x))
        x = self.conv2(self.relu(x))
        x = x + self.res_proj1(input)
        x = self.pool(self.relu(self.bn1(x)))

        # Residual Block 2
        input = x
        x = self.bn2(self.conv3(x))
        x = self.conv4(self.relu(x))
        x = x + self.res_proj2(input)
        x = self.pool(self.relu(self.bn2(x)))

        # Run through FC network
        x = torch.flatten(x, 1)
        x = self.relu(self.bnfc1(self.fc1(x)))
        x = self.relu(self.bnfc2(self.fc2(x)))
        x = self.fc3(x)
        return x


class CIFAR10Net_7(nn.Module):
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

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, return_features=False):
        # Run through CNN
        x = self.bn1(self.conv1(x))
        x = self.pool(self.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(self.relu(x))
        # Run through FC network
        x = torch.flatten(x, 1)
        x = self.relu(self.bnfc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bnfc2(self.fc2(x)))
        fc2_out = x.clone()
        x = self.dropout(x)
        x = self.fc3(x)
        if return_features:
            return x, fc2_out
        else:
            return x


class CIFAR10Net_optim(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 40, 4)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.fc1 = nn.Linear(40 * 6 * 6, 128)
        self.bnfc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bnfc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(p=0.46)

    def forward(self, x):
        # Run through CNN
        x = self.bn1(self.conv1(x))
        x = self.pool(self.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(self.relu(x))
        # Run through FC network
        x = torch.flatten(x, 1)
        x = self.relu(self.bnfc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bnfc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
