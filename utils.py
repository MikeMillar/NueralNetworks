import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from skimage.util.shape import view_as_windows
import numpy as np

def get_audio_filenames(dir):
    """
    Parses a directory and it's subdirectories for all audio files and
    adds their relative path to an array.

    Args:
        dir (string): The directory to start fetching audio files from.

    Returns:
        [string]: Array of string file paths of matching audio files.
    """

    print('Fetching audio file paths...')
    # Get all files in the directory
    dir_list = os.listdir(dir)
    # Initialize list
    filenames = []
    # Loop through all files
    for filename in dir_list:
        filepath = dir + filename
        # If file is a directory, need to recurse
        if os.path.isdir(filepath):
            ret = get_audio_filenames(filepath + '/')
            filenames += ret
            continue
        # Only care about files that end in au
        if filename.endswith('.au'):
            filenames.append(filepath)
    # Return filenames
    return filenames

def clean_model_dir(dir):
    """
    Utility function to clean up old model files that underperform.

    Args:
        dir (str): Path to the directory to clean up
    """
    model_files = os.listdir(dir)
    worst_acc = float('inf')
    worst_file = None
    while (len(model_files) > 5):
        for file in model_files:
            checkpoint = torch.load(file)
            checkpoint_acc = checkpoint['val_accuracy']
            if checkpoint_acc < worst_acc:
                worst_acc = checkpoint_acc
                worst_file = file
        os.remove(worst_file)
        worst_acc = float('inf')
        worst_file = None
        model_files = os.listdir(dir)

def create_image_window(image):
    """
    Utility function that takes an image file and creates subwindows of that image
    of size (224, 224). This size chose chosent since that was the size the 
    transfer learning model was originally trained on.

    Args:
        image (Tensor): Tensor image file

    Returns:
        (list): Returns list of tensor windows of the original image
    """
    image = np.transpose(np.array(image), (1, 0, 2))
    # Each image is a pytorch Tensor, need to create 224x224x3 image sections
    windows = view_as_windows(image, (224,224,3), 224)
    adjusted_windows = []
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            adjusted_windows.append(torch.tensor(np.transpose(windows[i,j,0], (2, 1, 0))))
    return adjusted_windows

def map_labels(labels):
    """
    Utility function to map the string labels to integers.

    Args:
        labels (list): List of labels to map

    Returns:
        (list): List of mapped labels
        (dict): Mapping of the labels from strings to ints
        (dict): Reverse mapping
    """
    unique_labels = list(set(labels))
    str_to_int = {string: i for i, string in enumerate(unique_labels)}
    int_to_str = {v: k for k, v in str_to_int.items()}
    return [str_to_int[string] for string in labels], str_to_int, int_to_str

def calculate_convolution_size(input_size, kernal_size, padding, stride):
    """
    Utility function to calculate the output size of a kernal operation.

    Args:
        input_size (tuple(int)): Original tensor size
        kernal_size (tuple(int)): Kernal size
        padding (int): How much to pad the original tensor
        stride (int): How much to move the kernal in each iteration

    Returns:
        (tuple(int)): Returns tuple of the new size
    """
    out_rows = math.floor(((input_size[0] + (2 * padding) - kernal_size[0]) / stride) + 1)
    out_cols = math.floor(((input_size[1] + (2 * padding) - kernal_size[1]) / stride) + 1)
    return (out_rows, out_cols)