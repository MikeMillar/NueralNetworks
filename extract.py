import os
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
import utils

# global variables
training_dir = 'data/train/'                    # training data directory
testing_dir = 'data/test/'                      # testing data directory
train = True                                    # True if we are extracting from training data
frameSize = 2048                                # ??
hopSize = 512                                   # ??


def save_spectrogram(title, y, sr, hop_length, y_axis='linear', save=False, show=False):
    """
    Creates a spectrogram from the audio input data and outputs the colormap of the spectrogram.
    If save is True, saves the file to graphs folder based on title's first word. If show
    is True, displays the spectrogram. If save is True, then show will not happen even if show
    is True.

    Args:
        title (str): Title of the generated spectrogram
        y (audio data): audio data extracted via librosa
        sr (float): Sample rate of the audio data
        hop_length (int): Window size of the audio data
        y_axis (str): Y axis scale factor
            - 'linear': linear y-axis scaling
            - 'log': log y-axis ascaling
        save (bool): True if graph should be saved to file
        show (bool): True if graph should be displayed (only if save = False)

    Returns:
        (Tensor): RGBA colormap tensor of shape (x, y, 4) where x and y are
            dependant on spectrogram size.
    """
    plt.figure(figsize=(10, 6))
    mesh = librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    colors = mesh.cmap(mesh.norm(mesh.get_array()))
    plt.title(title)
    if save:
        plt.savefig('graphs/stfts/' + title[:title.find(' ')] + '.png', bbox_inches='tight')
    elif show:
        plt.show()
    plt.close()
    return colors

def extract_and_plot(audio_data, frameSize, hopSize, title):
    """
    Uses librosa to extract the audio data and sampe rate, computes the STFT of the
    audio data and scales it. 

    Args:
        audio_data (str): File path to the audio file
        frameSize (int): ??
        hopSize (int): ??
        title (str): Base title of the spectrogram

    Returns:
        (audio data): Extracted and scaled STFT of the audio file
        (Tensor): RGBA colormap tensor of shape (x, y, 4) where x and y are
            dependant on the spectrograme size.
    """
    audio, sample_rate = librosa.load(audio_data)
    stft_audio = librosa.stft(audio, n_fft=frameSize, hop_length=hopSize)
    y_audio = np.abs(stft_audio) ** 2
    y_log_audio = librosa.power_to_db(y_audio)
    colors = save_spectrogram(title + ' log and y_axis log', y_log_audio, sample_rate, hopSize, y_axis='log')
    colors = colors[:, :, :3]
    scaler = MinMaxScaler()
    y_log_audio = scaler.fit_transform(y_log_audio)
    return y_log_audio, colors

def extract_and_save_colormap(basepath, fullpath):
    """
    Takes an audio file and extracts both the grayscale and RGBA colormaps
    of the audio data from a STFT spectrogram. Saves the data to the
    grayscale and rgba folders respectively as PyTorch tensors. These can
    be reloaded using the PyTorch load methods.

    Args:
        basepath (str): Relative directory of the audio files
        fullpath (str): Full relative filepath to the audio file

    Returns:
        None
    """
    # Extract only the filename
    filename = fullpath[fullpath.rfind('/')+1:]
    # Create the grayscale file path and rgb file path
    graypath = basepath + '_grays/' + filename[:filename.rfind('.')] + '_gray.pt'
    rgbpath = basepath + '_rgbas/' + filename[:filename.rfind('.')] + '_rgba.pt'
    # Extract the grayscale and colormap data
    gray_map, colormap = extract_and_plot(fullpath, frameSize, hopSize, filename[:filename.rfind('.')])
    # Convert maps to tensors
    gray_tensor = torch.tensor(gray_map)
    rgba_tensor = torch.tensor(colormap)
    # Save tensors
    torch.save(gray_tensor, graypath)
    torch.save(rgba_tensor, rgbpath)

if __name__ == '__main__':
    dir = testing_dir
    if train:
        dir = training_dir
    filepaths = utils.get_audio_filenames(dir)
    for file in filepaths:
        extract_and_save_colormap(dir, file)