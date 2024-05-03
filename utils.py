import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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