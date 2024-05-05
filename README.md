# Project 2 - Music Classification
In this project we will be classifying audio files into several different genres of music. We are given audio files for training which are sorted by genre already. We will use these audio files to extract features from them. These features will be used to classify the test audio files by genre based on a Logistic Regression Model we implement from scratch. This model will also utilize a gradient descent algorithm which will also be written from scratch. 

# Project Members
- Michael Millar: Sole developer

# Running the Application
The next few sections will step through the setup, configuration options, to actually running the application.

### Setup
1. Have python and pip installed on your computer.
2. Setup python virtual environment: \
    2a. Run command `python -m venv .venv` \
    2b. Activate virtual environment with `source .venv/bin/activate`
3. Use pip to install required libraries with `pip install -r requirements.txt`

### Configuration and Running the Application
The configuration of this project is done by modify variables and certain lines of code within the python files themselves. If you are interested in extracting features from music files, see `extract.py` in the file manifest. If you already have extracted feature data and want to run the model(s), see `my_mlp.py`, `my_cnn.py` and `transfer.py` in the file manifest. Note: Extraction methods for the data used in `my_mlp.py` are not included in this project, refer to the `extract.py` file from project 2.

# File Manifest
This project contains many parts which can be run independantly, or together. The following manifest will outline the different files in the application. This will include detailed descriptions of each file, which will detail their functions and use cases.

- `extract.py`: This file is used for the preprocessing and extraction of features in the audio files. The specified features and extracted and saved into an output file for later use. This file requires manually commenting/uncommenting of code, to determine which features are extracted. It has some additional parameters which can be modified:
    - training_dir: Directory of the training data.
    - testing_dir: Directory of the testing data.
    - train: Boolean to indicate if we are extracting data from the training or testing sets.
    - hop_size: Hop sized use by librosa during audio extraction.
    - frameSize: Frame sized use by librosa during audio extraction

    To use `extract.py`, follow the steps below:
    1. Change any of the global variables at the top of the file to desired values.
    2. Update either the train_dir or test_dir directory paths, then in the main method update which one is used. 
    3. Ensure you have both a `_grays/` and `_rgbas/` directories in your training or testing directories, as the script will not create them for you.
    4. Run the program with `python extract.py`
    5. Let program run, then check the output file.
- `my-mlp.py`: This file contains the multi-layer perceptron and the code to run it. It has several hyper-parameters that you can specify:
    - `learning_rate`: The rate at which the model progresses towards convergence during training.
    - `epsilon`: The error difference before early training termination.
    - `batch_size`: The number of rows of data to run on each batch of an epoch.
    - `max_epochs`: The maximum training iterations before termination.
    - `layer_sizes`: This is an array of the sizes of each layer. Currently only supports 3 values in the following order `[input, hidden, output]`.
    - `activation`: Activation function the model should use. This is restricted to valid PyTorch activation functions and is defautled to `ReLU()`.
    - `cost`: Cost/Loss function the model should use. This is restricted to valid PyTorch cost/loss functions and is defaulted to `CrossEntropyLoss()`.
    - `optimizer`: The optimizer function the model should use. This accepts two strings for this model: `SGD` or `Adam`, the optimizer is initialized with the model.
- `my-cnn.py`: This file contains the convolutional neural network and the code to run it. It has several hyper-parameters that you can specify:
    - `learning_rate`: The rate at which the model progresses towards convergence during training.
    - `epsilon`: The error difference before early training termination.
    - `batch_size`: The number of rows of data to run on each batch of an epoch.
    - `max_epochs`: The maximum training iterations before termination.
    - `layer_sizes`: This is an array of the sizes of each layer. Currently only supports 3 values in the following order `[input, hidden, output]`.
    - `activation`: Activation function the model should use. This is restricted to valid PyTorch activation functions and is defautled to `ReLU()`.
    - `cost`: Cost/Loss function the model should use. This is restricted to valid PyTorch cost/loss functions and is defaulted to `CrossEntropyLoss()`.
    - `optimizer`: The optimizer function the model should use. This accepts two strings for this model: `SGD` or `Adam`, the optimizer is initialized with the model.
- `transfer.py`: This model contains the code for the transfer learning model. The code is a bit of a mess, and has many design flaws and issues I didn't have time to resolve. It currently cant produce any outputs, so you would only be training a model and seeing that output to the console.
- `utils.py`: This file contains a handful of useful utility functions that are used throughout the program.
    - `clean_model_dir(dir)`: Utility function to clean up old model files that underperform.
    - `create_image_window(image)`: Utility function that takes an image file and creates subwindows of that image of size (224, 224). This size chose chosent since that was the size the transfer learning model was originally trained on.
    - `get_audio_filenames(dir)`: Fetches the audio filenames in the specified directory and subdirectories.
    - `map_labels(labels)`: Utility function to map the string labels to integers.
    - `calculate_convolution_size(input_size, kernel_size, padding, stride)`: Utility function to calculate the output size of a kernal operation.
- `graphs/`: Directory containing all of the statistical graphs used during the evaluation and reporting of this project.

# Kaggle Results
Best submission to the Kaggle competition was:
- May 4, 2024 with accuracy 48%