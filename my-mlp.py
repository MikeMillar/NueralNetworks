import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import utils

# CSV files
training_file = 'data/train/music_20_mfcc.csv'
testing_file = 'data/test/test_20_mfcc.csv'
# Parameters
batch_size = 60

class CustomDataset(Dataset):
    def __init__(self, data, labels, batch_size: int | None = 1):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __getitem__(self, index):
        row = self.data[index]
        label = self.labels[index]
        return row, label
    
    def __len__(self):
        return len(self.data)
    
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU(), cost=nn.CrossEntropyLoss(), optimizer='SGD', dropout_rate=0.2, learning_rate=1e-3):
        super(MLP, self).__init__()
        self.learning_rate = learning_rate
        self.activation = activation
        self.cost = cost
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            # if i < len(layer_sizes) - 2:
            #     layers.append(self.activation)
            #     layers.append(self.dropout)
        # self.linear_relu_stack = nn.Sequential(*layers)
        self.set_optimizer(optimizer)

    def forward(self, x):
        # return self.linear_relu_stack(x)
        z = x
        for i in range(len(self.layers)):
            layer = self.layers[i]
            z = self.activation(layer(z))
            if i < len(self.layers)-2:
                z = self.dropout(z)
        return z

    
    def set_optimizer(self, optimizer):
        # TODO: Add more optimizer options
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        

def load_data(file):
    return pd.read_csv(file, index_col=0)

def split_data(X):
    Y, classes = pd.factorize(X['label'])
    mapping = dict(zip(range(len(classes)), classes))
    X.drop('label', axis=1, inplace=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    return X_train, X_test, Y_train, Y_test, mapping

def normalize_data(X_train, X_test, Z):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Z = scaler.transform(Z)
    return X_train, X_test, Z

def load_split_normalize(train_file, test_file):
    # Load the data
    train = load_data(train_file)
    test = load_data(test_file)
    # Split the data
    X_train, X_test, Y_train, Y_test, mapping = split_data(train)
    # Normalize the data
    X_train, X_test, Z = normalize_data(X_train, X_test, test)
    # Transform data into datasets and dataloaders
    X_train = CustomDataset(X_train, Y_train)
    X_test = CustomDataset(X_test, Y_test)
    train_loader = DataLoader(X_train, batch_size=batch_size)
    test_loader = DataLoader(X_test, batch_size=batch_size)
    return train_loader, test_loader, Z, mapping

def train(dataloader, model):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = torch.tensor(np.array(X)).to(torch.float32).to(device), torch.tensor(np.array(y)).to(device)

        # Compute prediction error
        pred = model(X)
        loss = model.cost(pred, y)

        # Backpropagation
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

        loss, current = loss.item(), (batch+1) * len(X)
        print(f'loss: {loss:>7f}   [{current:>5d}/{size:>5d}]')

def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = torch.tensor(np.array(X)).to(torch.float32).to(device), torch.tensor(np.array(y)).to(device)
            pred = model(X)
            test_loss += model.cost(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

if __name__ == "__main__":
    # Load the data
    print('Preparing Data...')
    train_loader, test_loader, Z_loader, mapping = load_split_normalize(training_file, testing_file)

    # Define device, this will pick best available device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f'Using {device} device')

    # Define layer sizes
    # 1st element is input, last element is output
    # Each row contains 20 elements, output is 10 classes
    layer_sizes = [20, 15, 12, 10]

    # Define activation, cost functions
    activation = nn.ReLU()
    cost = nn.CrossEntropyLoss()

    # Create model
    model = MLP(layer_sizes, activation=activation, cost=cost, optimizer='SGD').to(device)
    print(model)

    # Define number of training epochs
    epochs = 100
    for t in range(epochs):
        print(f'Epoch {t+1}\n------------------------')
        train(train_loader, model)
        test(test_loader, model)