import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, inputSize, numClasses):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(inputSize, 50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50, numClasses)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

inputSize = 784
numClasses = 10
learningRate = 0.002
batchSize = 64
numEpochs = 5

trainDataset = datasets.MNIST(root = "dataset/", train=True, transform=transforms.ToTensor(), download = True)
trainLoader = DataLoader(dataset=trainDataset, batch_size = batchSize, shuffle=True)
testDataset = datasets.MNIST(root = "dataset/", train=False, transform=transforms.ToTensor(), download = True)
testLoader = DataLoader(dataset=testDataset, batch_size = batchSize, shuffle=True)

model = NN(inputSize=inputSize, numClasses=numClasses).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(numEpochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)
        
        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    
    numCorrect = 0
    numSamples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            numCorrect += (predictions == y).sum().item()
            numSamples += predictions.size(0)

        print("Got", numCorrect, "/", numSamples, "with accuracy", "{:.2f}".format(float(numCorrect) / float(numSamples) * 100))

    model.train()

check_accuracy(trainLoader, model)
check_accuracy(testLoader, model)
