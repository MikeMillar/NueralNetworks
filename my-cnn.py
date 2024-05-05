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

# Global variables
training_dir = 'data/train/_rgbas/'
testing_dir = 'data/train/_rgbas/'
output_path = 'data/result/test_out.csv'

criterion = nn.CrossEntropyLoss()
epochs = 50
test_size = 0.2
batch_size = 5


class SpectrogramDataset(Dataset):
    """
    Custom PyTorch dataset to dynamically load the image files.
    """
    def __init__(self, base_dir, paths, labels):
        """
        Initialize the dataset

        Args:
            base_dir (str): Base directory of the files
            paths (list(str)): List of filenames
            labels (list(int)): List of integer mapped labels
        """
        self.base_dir = base_dir
        self.paths = paths
        self.labels = labels

    def __getitem__(self, index):
        """
        Internal method used by PyTorch to train and validate the model,
        as well as produce predictions. Method dynamically loads the image
        files as needed to reduce memory requirements.

        Args:
            index (int): Index of data to retrieve

        Returns:
            image: Loaded image tensor
            label: Label of the image, if training set.
        """
        path = self.paths[index]
        image = torch.load(self.base_dir + path)
        # Trim images to ensure they are all the same size
        image = image[:,:1200,:]
        if self.labels == None:
            return image
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        """
        Outputs the total size of the dataset

        Returns
            (int): Size of the dataset
        """
        return len(self.paths)
    
class GenreClassifier(nn.Module):
    """
    Custom Genre Classification CNN
    """
    def __init__(self):
        """
        Initializes the model
        """
        super(GenreClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*112*112, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x): 
        """
        Internally used method of PyTorch for training, validation, and prediction.

        Args:
            x (Tensor): Tensor data
        """   
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def load_data(dir, hasLabels):
    """
    Utility method to load the filenames and labels of the image files
    
    Args:
        dir (str): String path to the data to load
        hasLabels (bool): Indication of if the data has labels to extract

    Returns:
        (list(str)): List of filenames
        (list(str)): List of labels, if applicable
    """
    files = os.listdir(dir)
    if not hasLabels:
        return files
    labels = []
    for file in files:
        labels.append(file[:file.find('.')])
    return files, labels
    
def load_split(dir, test_size, batch_size):
    """
    Utility method to load, map labels and split the data. The data is then
    added into custom PyTorch dataset and data loader objects.

    Args:
        dir (str): Directory of data to load
        test_size (float): Float value between 0 and 1 to indicate ratio of testing size.
        batch_size (int): How many images/labels to use each training cycle

    Returns:
        (dataloader): Training data loader
        (dataloader): Validation data loader
        (dict): Map of str to integer labels
        (dict): Reverse map    
    """
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

def train(dataloader, model, criterion, optimizer):
    """
    Method used to train an individual epoch for the MLP model.

    Args:
        dataloader (dataloader): Data loader of the training data
        model (nn.Module): ML model to train
        criterion (function): PyTorch cost/loss function
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        batch_loss = 0.0
        for i in range(len(y)):
            windows = utils.create_image_window(X[i])
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
        # print(f'loss: {(batch_loss / len(y)):>7f}   [{current:>5d}/{size:>5d}]')

def test(dataloader, model, criterion, accuracy):
    """
    Method used to validate an individual epoch for the MLP model.

    Args:
        dataloader (dataloader): Data loader of the validation data
        model (nn.Module): ML model to train
        criterion (function): PyTorch cost/loss function
        accuracy (function): Method to compute accuracy of model

    Returns:
        (float, float): Epoch accuracy and average loss
    """
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
                batch_loss += criterion(pred, targets).item()
                batch_acc += accuracy(pred, targets).item()
            test_loss += batch_loss / len(y)
            test_acc += batch_acc / len(y)
            
    test_loss /= num_batches
    test_acc /= num_batches
    print(f'Test Error: Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n')
    return test_acc, test_loss

def produce_output(dir, model, reverse_mapping, batch_size):
    """
    Method that loads the competition data, produces classification labels
    for it, and returns the audio file names and their predicted classifications.

    Args:
        dir (str): Directory path of the competition data
        model (nn.Module): CNN model
        reverse_mapping (dict): Dictionary map to map int labels back to strings
        batch_size (int): How many files to run at one time

    Returns:
        (dict): Returns dictionary of audio file names and the predicted classifications.
    """
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

train_loader, test_loader, mapping, reverse_map = load_split(training_dir, test_size, batch_size)

model = GenreClassifier()
model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
accuracy = MulticlassAccuracy(num_classes=10, average='micro').to(device=device)

for t in range(1, epochs+1):
    print(f'Epoch [{t}]\n-------------------------')
    train(train_loader, model, criterion, optimizer)
    test(test_loader, model, criterion, accuracy)

output = produce_output(testing_dir, model, reverse_map, batch_size)
out_df = pd.DataFrame(output)
out_df.to_csv(output_path, index=False)