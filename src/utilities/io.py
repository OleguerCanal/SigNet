import os

import pandas as pd
import torch

def read_data(device, experiment_id, source, data_folder="../data"):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    assert(source in ["random", "realistic"])
    path = os.path.join(data_folder, experiment_id)

    def csv_to_tensor(file):
        input_tensor = torch.tensor(pd.read_csv(file, header=None).values, dtype=torch.float)
        assert(not torch.isnan(input_tensor).any())
        assert(torch.count_nonzero(torch.sum(input_tensor, axis=1)) == input_tensor.shape[0])
        return input_tensor.to(device)

    train_input = csv_to_tensor(path + "/train_%s_input.csv"%source)
    train_baseline = csv_to_tensor(path + "/train_%s_baseline.csv"%source)
    train_label = csv_to_tensor(path + "/train_%s_label.csv"%source)

    val_input = csv_to_tensor(path + "/val_%s_input.csv"%source)
    val_baseline = csv_to_tensor(path + "/val_%s_baseline.csv"%source)
    val_label = csv_to_tensor(path + "/val_%s_label.csv"%source)

    return train_input, train_baseline, train_label, val_input, val_baseline, val_label

def read_methods_guesses(device, experiment_id, test_id, methods, data_folder="../data"):
    """Read one method guess from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        test_id (string): Full name of the test folder
        method (list): List of string with the methods to be analyzed
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    path = os.path.join(data_folder, experiment_id, test_id)

    def csv_to_tensor(file):
        input_tensor = torch.tensor(pd.read_csv(file, header=None).values, dtype=torch.float)
        assert(not torch.isnan(input_tensor).any())
        assert(torch.count_nonzero(torch.sum(input_tensor, axis=1)) == input_tensor.shape[0])
        return input_tensor.to(device)

    methods_guesses = []
    for method in methods:
        methods_guesses.append(csv_to_tensor(path + "/methods/%s_guess.csv"%(method)))
    
    label = csv_to_tensor(path + "/%s_label.csv"%(test_id))

    return methods_guesses, label
