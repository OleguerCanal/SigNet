import os

import pandas as pd
import torch

from utilities.data_partitions import DataPartitions


def csv_to_tensor(file, device):
    input_tensor = torch.tensor(pd.read_csv(
        file, header=None).values, dtype=torch.float)
    assert(not torch.isnan(input_tensor).any())
    # assert(torch.count_nonzero(torch.sum(input_tensor, axis=1))
    #        == input_tensor.shape[0])
    return input_tensor.float().to(device)


def read_data(device, experiment_id, source, data_folder="../data"):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    assert(source in ["random", "realistic", "mixed"])
    path = os.path.join(data_folder, experiment_id)

    train_input = csv_to_tensor(path + "/train_%s_input.csv" % source, device)
    train_baseline = csv_to_tensor(path + "/train_%s_baseline.csv" % source, device)
    train_label = csv_to_tensor(path + "/train_%s_label.csv" % source, device)

    train_data = DataPartitions(inputs=train_input,
                                prev_guess=train_baseline,
                                labels=train_label)

    val_input = csv_to_tensor(path + "/val_%s_input.csv" % source, device)
    val_baseline = csv_to_tensor(path + "/val_%s_baseline.csv" % source, device)
    val_label = csv_to_tensor(path + "/val_%s_label.csv" % source, device)

    val_data = DataPartitions(inputs=val_input,
                              prev_guess=val_baseline,
                              labels=val_label)

    return train_data, val_data


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

    methods_guesses = []
    for method in methods:
        methods_guesses.append(csv_to_tensor(
            path + "/methods/%s_guess.csv" % (method), device))

    label = csv_to_tensor(path + "/%s_label.csv" % (test_id), device)

    return methods_guesses, label


def read_test_data(device, experiment_id, test_id, data_folder="../data"):
    """Read one method guess from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        test_id (string): Full name of the test folder
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    path = os.path.join(data_folder, experiment_id, test_id)

    inputs = csv_to_tensor(path + "/%s_input.csv" % (test_id), device=device)
    label = csv_to_tensor(path + "/%s_label.csv" % (test_id), device=device)

    return inputs, label
