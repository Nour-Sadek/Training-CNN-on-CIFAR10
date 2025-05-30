# Training a Convolutional Neural Network on the CIFAR-10 dataset

After learning about CNNs from cs231n: Convolutional Neural Networks for Visual Recognition, lectures series 1-12 from 
2017 on YouTube: (https://youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&si=HtOL0A_Sv-wnu0VK), I followed 
through PyTorch's official documentation and practiced these following concepts concerning implementing deep learning 
models made up of Convolutional Networks by attempting to train a model on the CIFAR-10 dataset:
- Implementing a Convolutional Network from scratch by creating a class that inherits from torch.nn.Module and using 
regularization strategies such as L2 Regularization (weight decay), data augmentation (applying transforms on the 
training images such as ColorJitter and RandomRotation), and dropout.
- Using Kaiming Initialization and batch normalization to optimize the training process, as well as CrossEntropyLoss and 
Adam Optimization.
- Plotting the loss and accuracy of the training and validation data sets during training to monitor the training process.
- Utilizing the pytorch-Lightning and Optuna packages to tune specific hyperparameters and chose those that gave the 
best validation loss.
- Implementing Transfer Learning using the ResNet18 model and using its Convolutional Network as a feature extractor for 
training and optimizing the last fully connected layer on the CIFAR-10 dataset.
- Intuitively modifying the original, from-scratch model to see what kind of changes it would have on the training process.
- Performed Dimensionality Reduction Analysis by plotting a t-SNE plot after training was complete to cluster the test 
images based on the similarities of the features that are the output of the second-to-last fully connected layer.

# Objective

Train a deep learning model using pytorch on the CIFAR-10 dataset imported from torchvision datasets that is made up of 
45,000 training images, 5,000 validation images, and 10,000 test images, and try to optimize the model in an attempt to 
reduce overfitting and achieve a high validation accuracy.

# Methods

### Prepare the CIFAR-10 dataset for training

Loaded the CIFAR-10 dataset and divided the train set further between training and validation data sets.

    generator = torch.Generator().manual_seed(20)
    batch_size = 64
    
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

Calculated the mean and standard deviation for each dataset so that the input images would be normalized, as well as 
applied some data augmentation on the training set specifically for regularization.

    # Find the per channel mean and standard deviation of a given data set that was fed to a data loader
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

    # Generate the appropriate transform depending on data set, as well as its mean and std
    def apply_transforms(mean_std_tuple: tuple[torch.tensor, torch.tensor], train: bool) -> transforms:
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

### Prepare the deep learning model

Made a model class from scratch inheriting from torch.nn.Module, starting with a simple model with a couple of 
Convolutional Networks with pooling, followed by 3 Fully Connected layers (last layer outputs class scores). The ReLU 
activation function was used for non-linearity.

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

Kaiming Initialization of the weights with a normal distribution was used, with CrossEntropy as the loss function and 
Adam with default betas as the optimizer with a weight decay of 5e-4 and a learning rate of 0.001.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initializing the CIFAR10 model and applying Kaiming initialization using normal distribution
    model = CIFAR10Net()
    model.to(device)
    model.apply(initialize_using_norm_kaiming)

    # Determining the loss and optimization functions to be used during training
    criterion = nn.CrossEntropyLoss()
    # Values of lr and betas are already the default values of the hyperparameters, but I have them explicitly stated here
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)

After that, the model was trained for 100 epochs where the loss and accuracy of both training and validation datasets 
were recorded after every epoch.

    num_epochs = 100
    
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

### Hyperparameter tuning

To optimize model performance, specific hyperparameters were tuned using Optuna and pytorch-Lightning where the optimal 
values were chosen based on the combination that gave the lowest validation loss after 5 epochs over 30 trials.

    # Hyperparameter search space
    fc1_size = trial.suggest_categorical("fc1_size", [2 ** i for i in range(5, 9)])
    fc2_size = trial.suggest_categorical("fc2_size", [2 ** i for i in range(5, 9)])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.9)
    batch_size = trial.suggest_categorical("batch_size", [2 ** i for i in range(5, 9)])

### Transfer Training

Used the ResNet18 pre-trained model and trained it on CIFAR-10 dataset where the Convolutional Network was treated as a 
feature extractor since ImageNet and CIFAR-10 datasets are fairly similar and only the last Fully Connected layer (that 
connects to the classifications) was re-trained to fit CIFAR-10's 10 classes.

Note: transforms.Resize(224) was added to the list of transforms on the CIFAR-10 dataset as ResNet18 expects input images 
of sizes 224x224.

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

Training was done similarly on this network as the custom one, only that the last linear layer's parameters were allowed 
to be optimized during training.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)

### Dimensionality Reduction Analysis

The model is made up of a Convolutional Network followed by three Fully Connected layers. After the model was trained, 
the features of the 2nd Fully Connected layer (the one that connects to the layer that determines the classifications) 
were extracted for each image in the test set and the dimensionality reduction t-SNE technique was used to visualize 
this high dimensional feature space.

# Results

This is how the loss and validation of the train and validation test sets changed throughout training for 100 epochs for 
the first custom model that was a simple __Conv-ReLU-pool-Conv-ReLU-pool-FC1-ReLU-FC2-ReLU-FC3__:
![Image](https://github.com/user-attachments/assets/7f65f239-9efd-4e95-b717-43db79bb6551)

This is the typical look of these graphs when a model has been overfit on the training dataset where 
- the loss of the training keeps slowly going down, while the loss for the validation changes from going down to going up
- there is a noticeable gap in the accuracy of the test and validation sets where the accuracy of the validation set 
plateaus whereas that of the training keeps slowly going up.

After I have seen that, I thought of different changes that can be done to the model to prevent overfitting.

### Change #1
Since a simple Convolutional Network was used, overfitting could have been due to the fully connected layers, so I changed 
the model so that there are fewer neurons in each fully connected layer (CIFAR10Net_2)

The original model was: FC1 (124 neurons), FC2 (64 neurons), FC3 (10 neurons)
The proposed change is: FC1 (32 neurons), FC2 (16 neurons), FC3 (10 neurons)
![Image](https://github.com/user-attachments/assets/eb068b7e-aef2-43bf-84c7-3ffb7c55aa78)

This change still lead to the model over-fitting on the training dataset, although now it takes the model much longer to 
get better accuracy on the training dataset, with the accuracy of the validation dataset slowly going down over time and 
achieving an overall validation accuracy less than that of the original model.

### Change #2
Another change I wanted to try was to rather than decrease the number of neurons in each layer, I would decrease the 
number of fully connected layers (CIFAR10Net_3) and so the changed model would only have one fully connected layer where 
the features coming from the last pool layer immediately propagates to the fully connected layer made up of 10 neurons 
where each neuron represents a class score.
![Image](https://github.com/user-attachments/assets/25613889-cf89-4824-8b7d-c514174fa2fc)

Decreasing the number of fully connected layers didn't change the accuracy of the validation set and the model still 
over-fitting on the training dataset, although it seems like the new model needs more training epochs to reach a similar 
training accuracy as the first model.

### Change #3
Considering I started with a very small Convolutional Network, I wanted to test the effect of having a deeper convolutional 
network (4 ConvNets instead of 2) which in theory would lead to more over-fitting. I also increased the kernel size of the 
first ConvNets to 5x5 from 3x3 so that a broader look at the features would be taken in the beginning (CIFAR10Net_4).
![Image](https://github.com/user-attachments/assets/cd2a8697-969a-4968-971b-b78bb5110542)

This change lead to a drastic worsening and fluctuation of the accuracy of the training and validation sets. One thing of 
note is that the training and validation sets are mirroring each other in terms of their performance in accuracy. This 
zig-za shape might be an indication that the current learning rate, which has not been changed from the first model, might 
be too large for this model and so the model is struggling to converge.

### Change #4
This model was changed from the previous (Change #3) model by increasing the number of kernels for each convolutional 
layer and decreasing the number of fully connected layers by one (CIFAR10Net_5). I was expecting that change to lead to 
over-fitting but its effect on training would be unknown considering that the learning rate is still unchanged, and so 
we don't know if the current learning rate would be a better fit to this model or not.
![Image](https://github.com/user-attachments/assets/5325a165-5534-4e22-80f2-09f8645f58b7)

It did lead to the model over-fitting on the training dataset compared to the previous one (there is a gap between the two 
graphs, which wasn't present in the previous model). It also lead to better training and validation accuracy values even 
though still the zig-zag pattern is still present which means the learning rate is indeed too high for this model as well.

### Change #5
Generally adding residual blocks could stabilize training and allow the model to learn a simpler version by giving it the 
opportunity to learn the identity mapping when moving from one ConvNet to the next. To test that, I added two residual 
blocks (one connecting the input to the model to the output of the 2nd ConvNet and another that connects the input of the 
third Conv Net to the output of the 4th Conv Net) to the model corresponding to Change #3 (CIFAR10Net_6).
![Image](https://github.com/user-attachments/assets/ebd8f006-39c4-42c3-8355-723186338eb2)

That lead to better training compared, perhaps due to the opportunity of learning a simpler model, and that in turn lead 
to over-fitting, but again the zig-zag pattern remains which might mean that a more optimal learning rate could be used.

### Change #6
Going back to the original model made up of only 2 Convolutional Networks followed by 3 Fully Connected layers, I wanted 
to think of ways to prevent over-fitting and one of them is adding regularization strategies, and so I added dropout in 
the fully connected layers to see if that will lead to better training and less over-fitting (CIFAR10Net_7).
![Image](https://github.com/user-attachments/assets/e7423355-3b2e-4d7a-b89b-341eb44d6c90)

The model still over-fits the data (showcased by the training dataset having better accuracy and lower loss compared to the 
validation dataset), however validation loss no longer increases during training but decreases to a plateau, and the model 
does give an overall better validation accuracy. It seems like adding dropout enhanced the model, but it still needs 
improvement.

### Hyperparameter tuning
Considering that some of the changes done to the model weren't able to perform properly due to a badly chosen learning 
rate, it is obvious that the learning rate hyperparameter, among others, needed to be optimized. So, I sought to perform 
hyperparameter tuning over these hyperparameters as a start:
- learning rate (sample the log space between 1e-4 and 1e-1)
- weight decay (L2 regularization) (sample the log space between 5e-4 and 1e-1)
- dropout (sample the float space between 0.1 and 0.9)
- batch size (sample from the values 32, 64, 128, 256, 512)
- size of the first and second fully connected layers (sample from the values 32, 64, 128, 256, 512)

I only ran it once where the optimal combination of hyperparameters were chosen out of 30 trialed combos based off of the 
one that gave the lowest validation loss over 5 epochs. Then the model was trained using those optimized hyperparameters
(CIFAR10Net_optim).
![Image](https://github.com/user-attachments/assets/988b795f-6d19-47ae-a020-929a2769b4bf)

This model behaved very similarly to the previous one where only a dropout of 0.5 was used, with the only difference 
being that the loss of the training set reached lower values in less epochs.

### Transfer Learning
It is not very common for Convolutional deep learning models to be built from scratch, but rather it is more common to 
utilize transfer learning where models that were pre-trained on datasets such as ImageNet would be used and then slightly 
tweaked to fit the dataset at hand. Considering that the CIFAR-10 dataset is similar to that of ImageNet, I chose to use
the pre-trained ResNet18 mode, and rather than optimizing any of the later Convolutional layers, I instead focused on 
using the Convolutional layers as feature extractors and re-trained the model on the CIFAR-10 dataset where only the 
parameters of the last fully connected layer are optimized.
![Image](https://github.com/user-attachments/assets/32556348-90b2-4d57-b95f-ea8e916de79d)

Using transfer learning lead to the best performing model so far where even though it is still over-fitting on the training 
dataset, not as drastically at all as the first model, as well as it gave the highest validation accuracy of up to 80%. 
However, it still gives a sig-sag pattern which might mean that the learning rate value would need to be optimized.

### Dimensionality Reduction Analysis
I wanted to visualize the results of the training in some way, so I tried to visualize how similar are the features of 
the output of the second to last fully connected layer of the images of the test and train sets. To do that, the t-SNE dimensionality 
reduction technique was used. I used CIFAR10Net_7 after training it for 100 epochs. The colors correspond to the class 
that the model predicted each test/train image belongs to, and the points corresponding to those images that the model 
predicted its classification incorrectly have a black border around them.
This is the t-SNE plot on the test set:
![Image](https://github.com/user-attachments/assets/cebdb58c-2c3a-4c93-8abe-89ff3932c913)
This is the t-SNE plot on the train set:
![Image](https://github.com/user-attachments/assets/b9ba154d-65f9-4ae3-b18e-3921e8c52f23)

The images are clustered generally appropriately where the classes that correspond to animals are close together in the left 
half of the plot and those corresponding to vehicles are close together in the right half. Also the classes cluster nicely 
in both plots. There are two things that are of note looking at the t-SNE plot for the test images:
- There are images that were incorrectly predicted similarly to the points to which they cluster with, for example there 
are test images that were incorrectly predicted to be of class dog and that might be because their features cluster with 
those that are truly labeled as dog.
- There are images that are incorrectly predicted as a class that is different from the one that they cluster with based 
on their features. For example, there are a lot of test images that were classified as frog even though their features 
correlate with those whose true classification is bird.

There are possible reasons for the second observation, one being due to the inherit nature of what a t-SNE does on high 
dimensional data where data points might seem close in a 2D space, but are actually farther apart in higher dimensions. 
Another reason could be that the model misclassified them due to it overfitting to the training data.

There is also a mis-classification "blob" in the center that is most pronounced on the t-SNE plot for the training images. 
This might be due to potentially the model extracting similar features for these 10 classes due to an overlap in feature space 
between them, and so these points naturally get misclassified as the model is struggling to classify them to either of the 
classes confidently.

# Discussion and Next Steps


# Repository Structure

This repository contains:

    main.py: Implementation of a deep learning model from scratch where loss and accuracy of the train and validation sets 
    during training are plotted as well as performing Dimensionality Reduction Analysis by plotting a t-SNE plot.

    modified_models.py: Eight other models that were modified from the original one presented in main.py that were also 
    trained on the CIFAR-10 dataset to see if they had better performance.

    hyperparameter_tune.py: Performing hyperparameter tuning using pytorch-Lightning and Optuna to optimize the original 
    model in main.py, with the model with the optimal hyperparameters presented as CIFAR10Net_optim in modified_models.py.

    transfer_learning.py: Performing Transfer Learning using the pre-trained ResNet18 model. 
        
    functions.py: Four helper functions that are used across the python scripts.

    requirements.txt: List of required Python packages.

    CIFAR_7.pth: The saved model after it has been trained on the CIFAR-10 dataset for 100 epochs which used the values 
    of the optimal hyperparameters as obtained after one run of hyperparameter_tune.py, where dimensionality reduction 
    analysis was performed on it.

Python 3.12 version was used
