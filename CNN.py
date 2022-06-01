# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
from skorch.helper import SliceDataset


num_epochs = 4
num_classes = 10
learning_rate = 0.001
transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
download=True, transform=transform)
m = len(trainset)
train_data, val_data = random_split(trainset, [int(m - m * 0.2), int(m * 0.2)])
DEVICE = torch.device("cpu")
y_train = np.array([y for x, y in iter(train_data)])
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv_layer = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
             )
    self.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(8 * 8 * 64, 1000),
        nn.ReLU(inplace=True),
        nn.Linear(1000, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
    )
  def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
         # flatten
        x = x.view(x.size(0), -1)
         # fc layer
        x = self.fc_layer(x)
        return x

torch.manual_seed(0)
net = NeuralNetClassifier(
CNN,
max_epochs=1,
iterator_train__num_workers=0,
iterator_valid__num_workers=0,
lr=1e-3,
batch_size=64,
optimizer=optim.Adam,
criterion=nn.CrossEntropyLoss,
device=DEVICE
)
net.fit(train_data, y=y_train)
y_pred = net.predict(testset)
y_test = np.array([y for x, y in iter(testset)])
accuracy_score(y_test, y_pred)
plot_confusion_matrix(net, testset, y_test.reshape(-1, 1))
plt.show()

net.fit(train_data, y=y_train)
train_sliceable = SliceDataset(train_data)
scores = cross_val_score(net, train_sliceable, y_train, cv=5,
scoring="accuracy")