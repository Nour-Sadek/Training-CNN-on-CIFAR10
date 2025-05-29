# Training a Convolutional Neural Network on the CIFAR-10 dataset

After learning about CNNs from cs231n: Convolutional Neural Networks for Visual Recognition, lectures series 1-12 from 
2017 on YouTube: (https://youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&si=HtOL0A_Sv-wnu0VK), I followed 
through PyTorch's official documentation and practiced these following concepts concerning implementing deep learning 
models made up of Convolutional Networks by attempting to train a model on the CIFAR10 dataset:
- Implementing a Convolutional Network from scratch by creating a class that inherits from torch.nn.Module and using 
regularization strategies such as L2 Regularization (weight decay), data augmentation (applying transforms on the 
training images such as ColorJitter and RandomRotation), and dropout.
- Using Kaiming Initialization and batch normalization to optimize the training process, as well as CrossEntropyLoss and 
Adam Optimization.
- Plotting the loss and accuracy of the training and validation data sets during training to monitor the training process.
- Utilizing the pytorch-Lightning and Optuna packages to tune specific hyperparameters and chose those that gave the 
best validation loss.
- Implementing Transfer Learning using the ResNet18 model and using its Convolutional Network as a feature extractor for 
training and optimizing the last fully connected layer on the CIFAR10 dataset.
- Intuitively modifying the original, from-scratch model to see what kind of changes it would have on the training process.
- Performed Dimensionality Reduction Analysis by plotting a t-SNE plot after training was complete to cluster the test 
images based on the similarities of the features that are the output of the second-to-last fully connected layer.

# Objective






