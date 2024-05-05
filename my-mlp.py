import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import utils

# CSV files
training_file = 'data/train/music_20_mfcc.csv'
testing_file = 'data/test/test_20_mfcc.csv'
# Parameters
batch_size = 60
max_epochs = 100
learning_rate = 0.0001

# Define layer sizes
# 1st element is input, last element is output
# Each row contains 20 elements, output is 10 classes
# NOTE: Currently only accepts 3 values
layer_sizes = [20, 1028, 10]

# Define activation, cost, optimzer functions
activation = nn.ReLU()
cost = nn.CrossEntropyLoss()
optimizer = 'Adam'

class CustomDataset(Dataset):
    """
    Custom implementation dataset for use with our pandas dataframe
    """
    def __init__(self, data, labels):
        """
        Initializes the dataset with our data and labels.

        Args:
            data (dataframe): Dataframe of data
            labels (list): List of labels
        """
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        """
        Used by data loader to extract batches of data

        Args:
            index (int): Index of data

        Returns
            (arraylike, str): Returns arraylike of data and string label
        """
        row = self.data[index]
        label = self.labels[index]
        return row, label
    
    def __len__(self):
        """
        Outputs the total size of the dataset

        Returns
            (int): Size of the dataset
        """
        return len(self.data)
    
class MLP(nn.Module):
    """
    Custom implementation of our multi-layer perceptron using PyTorch
    """

    def __init__(self, layer_sizes, activation=nn.ReLU(), cost=nn.CrossEntropyLoss(), optimizer='SGD', dropout_rate=0.2, learning_rate=1e-3):
        """
        Initializes the network.

        Args:
            layer_sizes (arraylike): An arraylike object of layer sizes.
              This is currently hard coded to only accept total length of 3.
              The first element is the input size, second element is the
              hidden layer size, and the final element is the output size.
            activation (function): PyTorch activation function. Defaulted to ReLU()
            cost (function): PyTorch cost(loss) function. Defaulted to CrossEntropyLoss()
            optimizer (str): String representation of PyTorch optimizer to use. Valid
              options include: 'SGD', and 'Adam'
            dropout_rate (float): Float valued between 0 and 1, the dropout rate between layers.
            learning_rate (float): Learning rate used in parameter training.
        """
        super(MLP, self).__init__()
        # Initialize variables
        self.learning_rate = learning_rate
        self.activation = activation
        self.cost = cost
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList()
        # Create layers
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
        # Create optimizer
        self.set_optimizer(optimizer)

    def forward(self, x):
        """
        Internally used method of PyTorch for training, validation, and prediction.

        Args:
            x (Tensor): Tensor data
        """
        z = x
        for i in range(len(self.layers)):
            layer = self.layers[i]
            z = self.activation(layer(z))
            if i < len(self.layers)-2:
                z = self.dropout(z)
        return z

    
    def set_optimizer(self, optimizer):
        """
        Utility method to create the optimizer used for parameter training.

        Args:
            optimizer (str): String identifier of optimizer to use. Valid options
              are 'SGD' and 'Adam'.
        """
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        

def load_data(file):
    """
    Helper method to load pandas dataframe from file.

    Args:
        file (str): File path of dataframe

    Returns:
        (dataframe): Pandas dataframe
    """
    return pd.read_csv(file, index_col=0)

def split_data(X):
    """
    Method to split the data into training and validation steps. 
    Additionally performs a mapping of the labels to integers.

    Args:
        X (dataframe): Dataframe of data to split

    Returns:
        (dataframe): Feature training data
        (dataframe): Feature validation data
        (dataframe): Label training data
        (dataframe): Label validation data
        (dict): Mapping dictionary for mapping labels back to strings
    """
    Y, classes = pd.factorize(X['label'])
    mapping = dict(zip(range(len(classes)), classes))
    print(f'classes={classes}, mapping={mapping}')
    X.drop('label', axis=1, inplace=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    return X_train, X_test, Y_train, Y_test, mapping

def normalize_data(X_train, X_test, Z):
    """
    Helper method to transform the data into normalized form,
    with mean of zero, and standard deviation of 1.

    Args:
        X_train (dataframe): Training data
        X_test (dataframe): Validation data
        Z (dataframe): Competition data
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Z = scaler.transform(Z)
    return X_train, X_test, Z

def load_split_normalize(train_file, test_file):
    """
    Helper method to load, split and normalize the data. The method
    additionally converts the pandas dataframes into PyTorch datasets,
    and creates data loaders.

    Args:
        train_file (str): Path to training data
        test_file (str): Path to competition data

    Returns:
        (dataloader): Training data laoder
        (dataloader): Validation data loader
        (dataloader): Competition data loader
        (dict): Mapping of labels from str to int
    """
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
    """
    Method used to train an individual epoch for the MLP model.

    Args:
        dataloader (dataloader): Data loader of the training data
        model (nn.Module): ML model to train
    """
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
        # print(f'loss: {loss:>7f}   [{current:>5d}/{size:>5d}]')

def test(dataloader, model):
    """
    Method used to validate an individual epoch for the MLP model.

    Args:
        dataloader (dataloader): Data loader of the validation data
        model (nn.Module): ML model to train

    Returns:
        (float, float): Epoch accuracy and average loss
    """
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
    return correct, test_loss

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

    # Create model
    model = MLP(layer_sizes, activation=activation, cost=cost, optimizer=optimizer, learning_rate=learning_rate).to(device)
    print(model)

    loss = float('inf')
    acc = 0.0
    for t in range(max_epochs):
        print(f'Epoch {t+1}\n------------------------')
        train(train_loader, model)
        nAcc, nLoss = test(test_loader, model)
        if nAcc > 0.4 and nAcc > acc:
            acc = nAcc
            now = datetime.now()
            dt_string = now.strftime("%m-%d-%Y_%H-%M-%s")
            torch.save({
                'epoch': t+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'val_accuracy': acc,
            }, 'models/mlp/best_mlp_' + dt_string + '.pth')
        if abs(loss - nLoss) < 1e-6:
            print(f'Early termination')
            break
        loss = nLoss

    utils.clean_model_dir('models/mlp/')