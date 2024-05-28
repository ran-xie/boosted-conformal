import numpy as np
import random
import torch
import os, sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from real_datasets import datasets

base_dataset_path = '../real_datasets/'
verbose = True
def get_real_data(params,scale_X=True,scale_Y=True):
    # """ import and pre-process data
    # ----------
    # params : a dictionary with the following fields
    #    'dataset_name' : string, name of dataset
    #    'ratio'        : numeric, percentage of data used for training
    #    'seed'         : random seed
    # """

    # Extract main parameters
    dataset_name = params["data"]
    ratio_train = params["ratio"]
    seed = params["seed"]

    # Determines the size of test set
    test_ratio = 0.2

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize cuda if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load the data
    try:
        X, y = datasets.GetDataset(dataset_name, base_dataset_path)
        print("Loaded dataset '" + dataset_name + "'.")
        sys.stdout.flush()
    except:
        print("Error: cannot load dataset " + dataset_name)
        return

    # Dataset is divided into test and train data based on test_ratio parameter
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

    # Reshape the data
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    n_train = X_train.shape[0]

    # Print input dimensions
    print("Data size: train and calibration (%d, %d), test (%d, %d)" % (X_train.shape[0], X_train.shape[1],
                                                        X_test.shape[0], X_test.shape[1]))

    # Set seed for splitting the data into proper train and calibration
    np.random.seed(seed)
    idx = np.random.permutation(n_train)

    # Divide the data into proper training set and calibration set
    n_half = int(np.floor(n_train * ratio_train / 100.0))
    idx_train, idx_cal = idx[:n_half], idx[n_half:n_train]

    if(scale_X==True):
    # Zero mean and unit variance scaling of the train and test features
        scalerX = StandardScaler()
        scalerX = scalerX.fit(X_train[idx_train])
        X_train = scalerX.transform(X_train)
        X_test = scalerX.transform(X_test)
        
    if(scale_Y==True):
        # Scale the labels by dividing each by the mean absolute response
        mean_ytrain = np.mean(np.abs(y_train[idx_train]))
        y_train = np.squeeze(y_train)/mean_ytrain
        y_test = np.squeeze(y_test)/mean_ytrain
    else:
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
    return X_train[idx_train], y_train[idx_train], X_train[idx_cal], y_train[idx_cal], X_test, y_test

   