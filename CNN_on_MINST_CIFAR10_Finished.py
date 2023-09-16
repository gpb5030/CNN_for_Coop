"""
Title: CNN on MINST and CIFAR10
Developer: George Banacos
Date: 9/13/23
Description: This program uses pytorch to train a CNN on the MNIST and CIFAR10 
datasets. This program is inspired by Aladdin Persson's "Pytorch CNN example" 
https://www.youtube.com/watch?v=wnK3uWv_WkU
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CNN_MNIST(nn.Module):
    """
    This class defines the structure and forward propagation on the CNN model
    """
    def __init__(self, inChannels = 1, numClasses = 10):
        """
        This initializes the model
        @param inChannels: number of channels
        @param numClasses: The number of classifications in the dataset
        """
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels = 10, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2)) 
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(980, 490)
        self.fc2 = nn.Linear(490, 245)
        self.fc3 = nn.Linear(245, numClasses) 
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        """
        This function computes a forward propagation through the CNN. It starts
        at 28x28 then becomes 7x7 with 20 channels before fully connected
        layers
        @param x: The input image
        """
        x = F.relu(self.conv1(x)) #28x28
        x = self.pool(x) #14x14
        x = F.relu(self.conv2(x))
        x = self.pool(x) #7x7
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        x = self.fc3(x)
        return x

class CNN_CIFAR10(nn.Module):
    """
    This class defines the structure and forward propagation on the CNN model
    """
    def __init__(self, inChannels = 3, numClasses = 10):
        """
        This initializes the model
        @param inChannels: number of channels
        @param numClasses: The number of classifications in the dataset
        """
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels = 6, kernel_size=(3,3), stride=(1,1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels = 12, kernel_size=(3,3), stride=(1,1), padding=(0,0))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2)) 
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(1176, 490)
        self.fc2 = nn.Linear(490, 245)
        self.fc3 = nn.Linear(245, numClasses) 
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        """
        This function computes a forward propagation through the CNN. It starts
        at 32x32 then becomes 30x30 then 28x28 then 14x14 then 7x7 with 20 channels 
        before being flattened into a fully connected layers
        @param x: The input image
        """
        x = F.relu(self.conv1(x)) #32x32
        x = F.relu(self.conv2(x))
        x = self.pool(x) #14x14
        x = F.relu(self.conv3(x))
        x = self.pool(x) #7x7
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        x = self.fc3(x)
        return x


def check_accuracy(loader, model, device):
    """
    This function checks the accuracy of the model on both the training and test
    data sets. 
    @param Loader: The model loader
    @param model: The CNN model
    @param device: The device to load the model onto
    @return: None
    """
    if loader.dataset.train:
        print("Training accuracy:")
    else:
        print("Test accuracy")
    
    numCorrect = 0
    numSamples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device) 

            scores = model(x)
            _, predictions = scores.max(1)
            numCorrect += (predictions == y).sum().item() #.item to convert from tensor to int
            numSamples += predictions.size(0)

        print("Got", numCorrect, "/", numSamples, "with accuracy", "{:.2f}".format(float(numCorrect) / float(numSamples) * 100))

    model.train()


def main():
    """
    Inits the hyperparameters, training and test datasets, the device 
    and trains the model
    """
    trainDataset = None
    testDataset = None
    dataSetChoice = input("Enter \"M\" for MINST or enter \"C\" for CIFAR10: ")
    numEpochs = int(input("Enter number of epochs: "))
    if dataSetChoice == "M":
        learningRate = 0.003
        batchSize = 64
        trainDataset = datasets.MNIST(root = "dataset/", train=True, transform=transforms.ToTensor(), download = True)
        testDataset = datasets.MNIST(root = "dataset/", train=False, transform=transforms.ToTensor(), download = True)
    elif dataSetChoice == "C":
        learningRate = 0.002
        batchSize = 64
        trainDataset = datasets.CIFAR10(root = "dataset/", train=True, transform=transforms.ToTensor(), download = True)
        testDataset = datasets.CIFAR10(root = "dataset/", train=False, transform=transforms.ToTensor(), download = True)
    else:
        print("Bad choice!")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print("Device:", device)

    trainLoader = DataLoader(dataset=trainDataset, batch_size = batchSize, shuffle=True)
    testLoader = DataLoader(dataset=testDataset, batch_size = batchSize, shuffle=True)

    if dataSetChoice == "M":
        model = CNN_MNIST().to(device)
    else:
        model = CNN_CIFAR10(3).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(numEpochs):
        learningRate = learningRate * 0.985
        for batch_idx, (data, targets) in enumerate(trainLoader):
            data = data.to(device=device)
            targets = targets.to(device=device)
        
            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
        if (epoch % 2) == 0:
            print("Epoch", epoch, ":")
            check_accuracy(testLoader, model, device)

    check_accuracy(trainLoader, model, device)
    check_accuracy(testLoader, model, device)

main()