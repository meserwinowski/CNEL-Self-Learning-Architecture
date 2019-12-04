# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:00:33 2019

@author: meser
"""
# Standard Imports
import pickle
import sys
import time

# 3P Imports
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn
import torch.optim as optim
import torchvision as tv
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader

# Local Imports
sys.path.append("C://Users//meserwinowski//OneDrive - University of Florida//Graduate School//Research Assistant//CNEL Self-Learning Architecture//focus_of_attention")
import foa_image as foai
import foa_convolution as foac
import foa_saliency as foas

plt.rcParams.update(plt.rcParamsDefault)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Hyperparameters
classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
n_epochs = 1


def plot_confusion_matrix(cm, classes, title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('True label')
    plt.ylabel('Predicted label')


def run_foa(sample, prior, kernel, bbox=(28, 28)):

    # Import image
    img = foai.ImageObject(sample, rgb=False)

    # Generate Saliency Map
    foac.convolution(img, kernel, prior)

    # Bound and Rank the most Salient Regions of Saliency Map
    foas.salience_scan(img, rank_count=1, bbox_size=bbox)
    img.draw_image_patches()
    # img.plot_main()
    
    # Pad patches to be the same size (28x28)
    patch = img.patch_list[0]
    if (patch.shape != bbox):
        pad = np.zeros(bbox)
        pad[:patch.shape[0], :patch.shape[1]] = patch
        patch = pad

    return patch.reshape(1, bbox[0], bbox[1])


class MNIST(Dataset):
    """ MNIST Data Set """

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def shape(self):
        return [self.data.shape, self.targets.shape]


class Model(nn.Module):
    """ PyTorch Model Class """

    def __init__(self, cnn_layers=None, fc_layers=None):
        super(Model, self).__init__()

        # Network layer initialization
        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers

        # Generate Gaussian Blur Prior - Time ~0.0020006
        self.prior = foac.matlab_style_gauss2D((60, 60), 300)

        # Generate Gamma Kernel
        k = np.array([1, 9], dtype=float)
        mu = np.array([0.2, 0.5], dtype=float)
        self.kernel = foac.gamma_kernel(mask_size=(14, 14), k=k, mu=mu)

    def forward(self, x):

        # Apply Focus of Attention
        # x_crop = run_foa(x.numpy()[0], self.prior, self.kernel)
        # x = torch.tensor(x_crop).unsqueeze(0).to(device)
        x = x.unsqueeze(0).to(device)
            
        # Pass through convolutional network layers
        if (self.cnn_layers is not None):
            x = self.cnn_layers(x.type(torch.float))
            x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        if (self.fc_layers is not None):
            x = self.fc_layers(x.type(torch.float))

        return x


def train_model(train_loader, model, batch_size, criterion, optimizer):
    print("Training...")
    losses = []
    avg_loss = []
    loss_per_epoch = []
    total_train_time = 0
    for epoch in range(n_epochs):
        running_loss = 0.0  # Loss for each epoch

        # Iterate through each training sample
        start = time.time()
        for i, data in enumerate(train_loader, 0):

            # Get input training sample and send to device
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.to(device)
            # print(f"{i}: {inputs.shape}, {labels.shape} \n")

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Clear hyperparameter gradients
            optimizer.zero_grad()

            # Compute backpropgation gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Stats
            losses.append(loss.item())
            avg_loss.append(sum(losses) / len(losses))
            running_loss += loss.item()
            if (i % 500 == 0):
                print(f"Iteration: {i} | Average Loss: {avg_loss[-1]}")

        loss_per_epoch.append(loss.item())
        stop = time.time()
        train_time = stop - start
        total_train_time += train_time
        print(f"Training Epoch Time: {train_time}")
        print(f"Epoch: {epoch}/{n_epochs - 1}")

    return model, avg_loss, total_train_time


def test_model(test_loader, model):
    print("Testing...")
    predicted = np.array([])
    expected = np.array([])

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        # inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        expected = np.append(expected, labels.cpu())
        predicted = np.append(predicted, pred.item())

    print(f"True: {labels.item()} | Predicted: {predicted[-1]}")

    # accuracy = accuracy_score(expected, predicted)
    return expected, predicted


def evaluation(dataloader, model):

    # Function to calculate the accuracy
    total, correct = 0, 0
    for data in dataloader:
        # Get the input and labels from data
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total += labels.size(0)

        # Calculate the accuracy
        count = np.where(labels - outputs < 0.5, 1, 0).sum()
        correct += count

    return (100 * correct / total)


def full_test(train_set, test_set, batch_size, model, criterion, optimizer):

    # Create data loaders
    print("Create Dataloaders")
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)

    # Training
    model, avg_loss, train_time = train_model(train_loader, model, batch_size, criterion, optimizer)
    print(f"Total Training Time: {train_time}")

    # Test Model
    expected, predicted = test_model(test_loader, model)

    return avg_loss, expected, predicted


if __name__ == "__main__":

    # %% Load MNIST Input data
    tv.datasets.MNIST("./", train=True, download=True)
    mnist_data = torch.load("./MNIST/processed/training.pt")
    mnist_train_data = mnist_data[0].data
    mnist_train_labels = mnist_data[1].data
    mnist_data = torch.load("./MNIST/processed/test.pt")
    mnist_test_data = mnist_data[0].data
    mnist_test_labels = mnist_data[1].data

    # %% Define Model
    input_size = 60
    batch_size = 1

    # Model 1
    cnn_layers_1 = nn.Sequential(
        nn.Conv2d(1, 7, kernel_size=3),
        nn.MaxPool2d(2, stride=3),
        nn.LeakyReLU()
    )
        
    fc_layers_1 = nn.Sequential(
        nn.Linear(567, 10),
        nn.LeakyReLU(),
    )

    # Model 2
    cnn_layers_2 = nn.Sequential(
        nn.Conv2d(1, 14, kernel_size=3),
        nn.MaxPool2d(2, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(14, 7, kernel_size=1),
        nn.MaxPool2d(2, stride=2),
        nn.LeakyReLU()
    )
        
    fc_layers_2 = nn.Sequential(
        nn.Linear(252, 10),
        nn.LeakyReLU(),
    )
    
    # Model 3
    cnn_layers_3 = nn.Sequential(
        nn.Conv2d(1, 30, kernel_size=3),
        nn.MaxPool2d(2, stride=2),
        nn.LeakyReLU(),
        nn.Conv2d(30, 2, kernel_size=3),
        nn.MaxPool2d(2, stride=2),
        nn.LeakyReLU()
    )
        
    fc_layers_3 = nn.Sequential(
        nn.Linear(338, 10),
        nn.LeakyReLU(),
    )

    model = Model(cnn_layers_3, fc_layers_3).to(device)  # Instantiate model
    criterion = nn.CrossEntropyLoss()  # Loss Function
    optimizer = optim.Adam(model.parameters())  # Optimizer function and learning rate

    # %% Run
    avg_losses = []
    predicted = []

    # Create Training and Test Data
    filename = "./ClutteredMNIST/train_cnn.pickle"
    with open(filename, 'rb') as infile:
        data = pickle.load(infile)

    train_set_np = data[0]
    train_labels_np = data[1].astype(np.int64)

    filename = "./ClutteredMNIST/test_cnn.pickle"
    with open(filename, 'rb') as infile:
        data = pickle.load(infile)

    test_set_np = data[0]
    test_labels_np = data[1]

    # Create data set
    train_set = MNIST(train_set_np, train_labels_np)
    test_set = MNIST(test_set_np, test_labels_np)

    # Run test
    avg_loss, expec, pred = full_test(train_set, test_set, batch_size,
                                      model, criterion, optimizer)

    # %% Display Data
    cnf_matrix = confusion_matrix(pred, expec)
    accuracy = accuracy_score(expec, pred)
    print(f"FoA CNN Test Accuracy: {accuracy * 100}")

    test = 3
    plt.figure()
    cnf_matrix_title = f"FoA CNN Confusion Matrix {test}"
    plot_confusion_matrix(cnf_matrix, classes=classes, title=cnf_matrix_title)
    plt.plot()

    # Plot Loss Curve
    plt.figure()
    avg_loss = np.array(avg_loss)
    n_avg_loss = avg_loss / avg_loss.max()
    plt.plot(n_avg_loss, linewidth=2)
    print(f"Final loss: {n_avg_loss[-1]:.4f}")
    plt.title(f"Learning Curve - {criterion}; Epochs: {n_epochs}")
    plt.xlabel("Iterations")
    plt.ylabel("Averaged Loss")
    plt.show()
