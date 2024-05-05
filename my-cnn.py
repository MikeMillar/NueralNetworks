import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import utils

class SpectrogramDataset(Dataset):
    def __init__(self, base_dir, paths, labels):
        self.base_dir = base_dir
        self.paths = paths
        self.labels = labels

    def __getitem__(self, index):
        path = self.paths[index]
        image = torch.load(self.base_dir + path)
        # Trim images to ensure they are all the same size
        image = image[:,:1200,:]
        if self.labels == None:
            return image
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        return len(self.paths)
    
class GenreClassifier(nn.Module):
    def __init__(self):
        super(GenreClassifier, self).__init__()
        # self.network = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2), # Output size 64 x 112 x 112

        #     nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2), # output size 128 x 56 x 56

        #     nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=(3,3), stride=(3,3), padding=(3,3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2,2), # output size 256 x 28 x 28

        #     nn.Flatten(),
        #     nn.Linear(25600, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10)
        # )
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(64*56*56, 512)
        self.fc1 = nn.Linear(32*112*112, 256)
        self.fc2 = nn.Linear(256, 10)
        # self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # return self.network(x.float())        
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        # x = self.fc3(x)
        return x
    
def load_data(dir, hasLabels):
    files = os.listdir(dir)
    if not hasLabels:
        return files
    labels = []
    for file in files:
        labels.append(file[:file.find('.')])
    return files, labels
    
def load_split(dir, test_size, batch_size):
    # Load the data from file
    files, labels = load_data(dir, True)
    labels, mapping, reverse_map = utils.map_labels(labels)
    # Split the data into train/validation
    train_files, test_files, train_labels, test_labels = train_test_split(files, labels, test_size=test_size, stratify=labels)
    # Create the datasets
    train_dataset = SpectrogramDataset(dir, train_files, train_labels)
    test_dataset = SpectrogramDataset(dir, test_files, test_labels)
    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # Return loaders
    return train_loader, test_loader, mapping, reverse_map

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def train(dataloader, model, criterion, optimizer, accuracy):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_loss = 0.0
        for i in range(len(y)):
            windows = utils.create_image_window(X[i])
            # data = torch.tensor(np.array(windows)).float().to(device=device)
            data = torch.FloatTensor(np.array(windows)).to(device=device)
            targets = torch.tensor(np.array([y[i]] * len(data))).to(device=device)

            # Compute prediction error
            pred = model(data)
            loss = criterion(pred, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss += loss.item()

        current = (batch+1) * len(X)
        print(f'loss: {(batch_loss / len(y)):>7f}   [{current:>5d}/{size:>5d}]')

def test(dataloader, model, criterion, accuracy):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            batch_loss = 0
            batch_acc = 0
            for i in range(len(y)):
                windows = utils.create_image_window(X[i])
                data = torch.FloatTensor(np.array(windows)).to(device=device)
                targets = torch.tensor(np.array([y[i]] * len(data))).to(device=device)
                pred = model(data)
                # print(pred, targets)
                batch_loss += criterion(pred, targets).item()
                batch_acc += accuracy(pred, targets).item()
            test_loss += batch_loss / len(y)
            test_acc += batch_acc / len(y)
            
    test_loss /= num_batches
    test_acc /= num_batches
    print(f'Test Error: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    return test_acc, test_loss

def produce_output(dir, model, reverse_mapping, batch_size):
    # Load the data
    files = load_data(dir, False)
    data = SpectrogramDataset(dir, files, None)
    loader = DataLoader(data, batch_size=batch_size)

    # Produce the predictions
    predictions = []
    with torch.no_grad():
        for batch, X in enumerate(loader):
            for i in range(X.shape[0]):
                windows = utils.create_image_window(X[i])
                data = torch.FloatTensor(np.array(windows)).to(device=device)
                
                pred = model(data)
                predIdx = torch.argmax(pred, dim=1)
                ensemble_val = torch.mode(predIdx, 0).values.item()
                predictions.append(ensemble_val)

    # Map integer predictions back to strings
    pred_str = [reverse_map[x] for x in predictions]
    # Fix output file names
    renamed = [x[:x.find('_')] + '.au' for x in files]
    return {
        'id': renamed,
        'class': pred_str
    }

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, mapping, reverse_map = load_split('data/train/_rgbas/', 0.2, 5)


model = GenreClassifier()
model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
accuracy = MulticlassAccuracy(num_classes=10, average='micro').to(device=device)
criterion = nn.CrossEntropyLoss()

epochs = 1
for t in range(1, epochs+1):
    print(f'Epoch [{t}]\n-------------------------')
    train(train_loader, model, criterion, optimizer, accuracy)
    test(test_loader, model, criterion, accuracy)

output = produce_output('data/test/_rgbas/', model, reverse_map, 5)
out_df = pd.DataFrame(output)
out_df.to_csv('data/result/test_out.csv', index=False)