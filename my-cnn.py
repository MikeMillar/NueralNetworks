import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

import utils

class SpectrogramDataset(Dataset):
    def __init__(self, base_dir, paths, labels):
        self.base_dir = base_dir
        self.paths = paths
        self.labels = labels

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = torch.load(self.base_dir + path)
        # Trim images to ensure they are all the same size
        image = image[:,:1200,:]
        return image, label
    
    def __len__(self):
        return len(self.paths)
    
class GenreClassifier(nn.Module):
    def __init__(self):
        super(GenreClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output size 64 x 112 x 112

            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output size 128 x 56 x 56

            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(3,3), padding=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output size 256 x 28 x 28

            nn.Flatten(),
            nn.Linear(25600, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.network(x)        
    
def load_data(dir):
    files = os.listdir(dir)
    labels = []
    for file in files:
        labels.append(file[:file.find('.')])
    return files, labels
    
def load_split(dir, test_size, batch_size):
    # Load the data from file
    files, labels = load_data(dir)
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

def train(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        for i in range(len(y)):
            windows = utils.create_image_window(X[i])
            data = torch.tensor(np.array(windows)).float().to(device=device)
            targets = torch.tensor(np.array([y[i]] * len(data))).to(device=device)

            # Compute prediction error
            pred = model(data)
            loss = criterion(pred, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss, current = loss.item(), (batch+1) * len(X)
            print(f'loss: {loss:>7f}   [{current:>5d}/{size:>5d}]')

def test(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            for i in range(len(y)):
                windows = utils.create_image_window(X[i])
                data = torch.tensor(np.array(windows)).float().to(device=device)
                targets = torch.tensor(np.array([y[i]] * len(data))).to(device=device)
                pred = model(X)
                test_loss += criterion(pred, targets).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    return correct, test_loss

input_size = (224, 224)
kernal_size = (3, 3)
padding = 7
stride = 2
rows, cols = utils.calculate_convolution_size(input_size, kernal_size, padding, stride)
print(f'Starting size={input_size}, k_size={kernal_size}, pad={padding}, stride={stride}')
print(f'Result size={(rows, cols)}')

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loader, mapping, reverse_map = load_split('data/train/_rgbas/', 0.2, 1)


model = GenreClassifier()
model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 5
for t in range(1, epochs+1):
    print(f'Epoch [{t}]\n-------------------------')
    train(train_loader, model, criterion, optimizer)
    test(test_loader, model, criterion)