import os
import torch
import torch.nn as nn
import torchvision
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader, Dataset
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
    def __init__(self, backbone, load_pretrained, learning_rate):
        super(GenreClassifier, self).__init__()
        self.backbone = backbone
        self.learning_rate = learning_rate
        self.pretrained_model = None
        self.classifier_layers = []
        self.new_layers = []

        # Define model
        if backbone == "resnet18":
            if load_pretrained:
                self.pretrained_model = torchvision.models.resnet18(
                    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                )
            else:
                self.pretrained_model = torchvision.models.resnet18(weights=None)

            # Replace classification layer
            self.classifier_layers = [self.pretrained_model.fc]
            self.pretrained_model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
            self.new_layers = [self.pretrained_model.fc]
        elif backbone == "resnet50":
            if load_pretrained:
                self.pretrained_model = torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                )
            else:
                self.pretrained_model = torchvision.models.resnet50(weights=None)

            # Replace classification layer
            self.classifier_layers = [self.pretrained_model.fc]
            self.pretrained_model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
            self.new_layers = [self.pretrained_model.fc]
        elif backbone == "resnet152":
            if load_pretrained:
                self.pretrained_model = torchvision.models.resnet152(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                )
            else:
                self.pretrained_model = torchvision.models.resnet152(weights=None)

            # Replace classification layer
            self.classifier_layers = [self.pretrained_model.fc]
            self.pretrained_model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
            self.new_layers = [self.pretrained_model.fc]

        self.register_buffer('train_metrics', torch.tensor([float('inf'), 0.0]))
        self.register_buffer('val_metrics', torch.tensor([float('inf'), 0.0]))

    def update_metrics(self, run_type, loss, accuracy):
        metrics = self.train_metrics if run_type == "train" else self.val_metrics
        if loss is not None:
            metrics[0] = loss
        if accuracy is not None:
            metrics[1] = accuracy

    def get_metrics(self, run_type):
        metrics = self.train_metrics if run_type == "train" else self.val_metrics
        return dict(zip(['loss', 'accuracy'], metrics.tolist()))

    def set_trainable(self, setting):
        # Freeze all layers
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
        # Unfreeze desired layers
        if setting == 'N':      # Unfreeze all newly created layers
            for layer in self.new_layers:
                for p in layer.parameters():
                    p.requires_grad = True
        elif setting == 'C':    # Unfreeze pre-existing layers
            for layer in self.classifier_layers:
                for p in layer.parameters():
                    p.requires_grad = True
        else:                   # Unfreeze all layers
            for p in self.pretrained_model.parameters():
                p.requires_grad = True

    def get_optimizer_params(self):
        options = []
        layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
        lr = self.learning_rate
        for layer in reversed(layers):
            options.append(
                {
                    "params": getattr(self.pretrained_model, layer).parameters(),
                    "lr": lr,
                }
            )
            lr /= 3.0
        return options

    def forward(self, x):
        return self.pretrained_model(x)
    
    def train_epoch(self, train_loader, optimizer, epoch):
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        num_batches = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            for i in range(len(y)):
                windows = utils.create_image_window(X[i])
                data = torch.tensor(np.array(windows)).float().to(device="cpu")
                targets = torch.tensor(np.array([y[i]] * len(data))).to(device="cpu")
                pred = self(data)
                loss = criterion(pred, targets)

                running_loss, num_batches = running_loss + loss.item(), num_batches + 1
                loss.backward()
                optimizer.step()
        print(f'[{epoch}] Train Loss: {running_loss / num_batches:0.5f}')
        return running_loss / num_batches
    
    def evaluate(self, loader, metric, epoch, run_type):
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0
        for X, y in loader:
            for i in range(len(y)):
                windows = utils.create_image_window(X[i])
                data = torch.tensor(np.array(windows)).float().to(device="cpu")
                targets = torch.tensor(np.array([y[i]] * len(data))).to(device="cpu")
                pred = self(data)
                loss = criterion(pred, targets)

                running_loss = running_loss = loss.item()
                num_batches += 1
                running_accuracy += metric(pred, targets).item()

        print(f'[{epoch}] {run_type} Loss: {running_loss / num_batches:0.5f}, Accuracy: {running_accuracy / num_batches:0.5f}')
        return running_loss / num_batches, running_accuracy / num_batches
    
    def train_and_save(self, train_loader, test_loader, accuracy, optimizer, 
              scheduler, epochs, filename):
        best_accuracy = self.get_metrics('val')['accuracy']
        for epoch in range(1, epochs + 1):
            # Perform 1 epoch of training
            self.train()
            self.train_epoch(train_loader, optimizer, epoch)

            # Evaluate training
            with torch.inference_mode():
                train_loss, train_acc = self.evaluate(train_loader, accuracy, epoch, "Train")
                # training_run.train_loss.append(train_loss)
                # training_run.train_accuracy.append(train_acc)

            # Evaluate accuracy on the validation dataset
            self.eval()
            with torch.inference_mode():
                val_loss, val_acc = self.evaluate(test_loader, accuracy, epoch, "Validation")
                # training_run.val_loss.append(val_loss)
                # training_run.val_accuracy.append(val_acc)
                if val_acc > best_accuracy:
                    # Model is better than previous best models, save model checkpoint
                    self.update_metrics('train', train_loss, train_acc)
                    self.update_metrics('val', val_loss, val_acc)
                    torch.save(self.state_dict(), filename)
                    best_accuracy = val_acc
            
            scheduler.step()

def load_data(dir):
    files = os.listdir(dir)
    labels = []
    for file in files:
        labels.append(file[:file.find('.')])
    return files, labels
    
def load_split(dir, test_size, batch_size):
    # Load the data from file
    files, labels = load_data(dir)
    labels, map, reverse_map = utils.map_labels(labels)
    # Split the data into train/validation
    train_files, test_files, train_labels, test_labels = train_test_split(files, labels, test_size=test_size, stratify=labels)
    # Create the datasets
    train_dataset = SpectrogramDataset(dir, train_files, train_labels)
    test_dataset = SpectrogramDataset(dir, test_files, test_labels)
    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # Return loaders
    return train_loader, test_loader, map, reverse_map

train_loader, test_loader, map, reverse_map = load_split('data/train/_rgbas/', 0.2, 1)

model = GenreClassifier('resnet18', True, 0.01)
model.to(device="cpu")

optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)
accuracy = MulticlassAccuracy(num_classes=10, average='micro').to(device="cpu")

print('Initial accuracy of resnet18 with classification layer replaced')
model.eval()
model.evaluate(test_loader, accuracy, 0, 'Val')

# Freeze all weights except newly added layer
model.set_trainable('N')

best_test_accuracy = 0.0

model.train_and_save(train_loader, test_loader, accuracy, optimizer,
            scheduler, 16, 'models/transfer/resnet18.pt')

print('Done with feature extraction for resnet18.')
best_val_accuracy = model.get_metrics('va')['accuracy']
print(f'resnet18 best val accuracy after feature extraction: {best_val_accuracy}')

print('Running fine-tunning')

# Set all weights trainable
model.set_trainable('A')

optimizer = torch.optim.Adam(model.get_optimizer_params(), lr=1e-8)
# Every 2 steps, reduce LR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

model.train_and_save(train_loader, test_loader, accuracy, optimizer, scheduler,
            8, 'models/transfer/resnet18.pt')

best_val_accuracy = model.get_metrics('val')['accuracy']
print(f'resnet18 best val accuracy after fine-tuning: {best_val_accuracy}')