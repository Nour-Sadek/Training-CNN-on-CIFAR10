import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from functions import mean_std_per_channel
from functions import apply_transforms

generator = torch.Generator().manual_seed(20)
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Finding out the mean and standard deviation of the CIFAR10 dataset so that I
# would normalize based on that
transform = transforms.ToTensor()

# Get the train and test datatsets
trainset = torchvision.datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=False,
                                        transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=False,
                                       transform=transforms.ToTensor())

# Divide the trainset further into trainset and validation set
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
trainset.dataset.transform = apply_transforms(trainset, train_mean_std, train=True)
valset.dataset.transform = apply_transforms(valset, val_mean_std, train=False)
testset.transform = apply_transforms(testset, test_mean_std, train=False)


class LitCIFAR10Net(pl.LightningModule):
    def __init__(self, fc1_size=124, fc2_size=64, lr=1e-3, weight_decay=1e-4, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()

        # CNN layers
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 40, 4)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool = nn.MaxPool2d(2, 2)

        # FC layers (parameterized)
        self.fc1 = nn.Linear(40 * 6 * 6, self.hparams.fc1_size)
        self.bnfc1 = nn.BatchNorm1d(self.hparams.fc1_size)
        self.fc2 = nn.Linear(self.hparams.fc1_size, self.hparams.fc2_size)
        self.bnfc2 = nn.BatchNorm1d(self.hparams.fc2_size)
        self.fc3 = nn.Linear(self.hparams.fc2_size, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=self.hparams.dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.bnfc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bnfc2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )


def objective(trial):
    # Hyperparameter search space
    fc1_size = trial.suggest_categorical("fc1_size", [2 ** i for i in range(5, 9)])
    fc2_size = trial.suggest_categorical("fc2_size", [2 ** i for i in range(5, 9)])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.9)
    batch_size = trial.suggest_categorical("batch_size", [2 ** i for i in range(5, 9)])

    # Data
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = LitCIFAR10Net(fc1_size=fc1_size, fc2_size=fc2_size, lr=lr, weight_decay=weight_decay, dropout=dropout)

    # Trainer
    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",
        logger=TensorBoardLogger("optuna_logs", name="cifar10"),
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_loss"].item()


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Trained for 5 epochs max
print("Best hyperparameters:", study.best_trial.params)

# Best hyperparameters: {'fc1_size': 128, 'fc2_size': 128, 'lr': 0.00216129196936161,
# 'weight_decay': 0.0009249448187052393, 'dropout': 0.4568220239272306, 'batch_size': 128} after 30 trials
# Val loss was 0.8492231369018555
