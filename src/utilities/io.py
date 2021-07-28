import pandas as pd
import torch

def read_data(device, data_folder="../data"):
    train_input = torch.tensor(pd.read_csv(
        data_folder + "/train_input_w01.csv", header=None).values, dtype=torch.float)
    train_input = train_input.to(device)
    train_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/train_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    train_guess_0 = train_guess_0.to(device)
    train_label = torch.tensor(pd.read_csv(
        data_folder + "/train_label_w01.csv", header=None).values, dtype=torch.float)
    train_label = train_label.to(device)

    val_input = torch.tensor(pd.read_csv(
        data_folder + "/validation_input_w01.csv", header=None).values, dtype=torch.float)
    val_input = val_input.to(device)
    val_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/validation_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    val_guess_0 = val_guess_0.to(device)
    val_label = torch.tensor(pd.read_csv(
        data_folder + "/validation_label_w01.csv", header=None).values, dtype=torch.float)
    val_label = val_label.to(device)
    return train_input, train_guess_0, train_label, val_input, val_guess_0, val_label


def read_data_realistic(device, data_folder="../data/realistic_data/train_default"):
    train_input = torch.tensor(pd.read_csv(
        data_folder + "/realistic_train_input.csv", header=None).values, dtype=torch.float)
    train_input = train_input.to(device)
    train_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/realistic_train_baseline_JS.csv", header=None).values, dtype=torch.float)
    train_guess_0 = train_guess_0.to(device)
    train_label = torch.tensor(pd.read_csv(
        data_folder + "/realistic_train_label.csv", header=None).values, dtype=torch.float)
    train_label = train_label.to(device)

    val_input = torch.tensor(pd.read_csv(
        data_folder + "/realistic_validation_input.csv", header=None).values, dtype=torch.float)
    val_input = val_input.to(device)
    val_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/realistic_validation_baseline_JS.csv", header=None).values, dtype=torch.float)
    val_guess_0 = val_guess_0.to(device)
    val_label = torch.tensor(pd.read_csv(
        data_folder + "/realistic_validation_label.csv", header=None).values, dtype=torch.float)
    val_label = val_label.to(device)
    return train_input, train_guess_0, train_label, val_input, val_guess_0, val_label

def read_data_realistic_larger(device, data_folder="../data/realistic_data"):
    train_input = torch.tensor(pd.read_csv(
        data_folder + "/train_more_sigs/larger_realistic_train_input.csv", header=None).values, dtype=torch.float)
    train_input = train_input.to(device)
    train_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/train_more_sigs/larger_realistic_train_baseline_JS.csv", header=None).values, dtype=torch.float)
    train_guess_0 = train_guess_0.to(device)
    train_label = torch.tensor(pd.read_csv(
        data_folder + "/train_more_sigs/larger_realistic_train_label.csv", header=None).values, dtype=torch.float)
    train_label = train_label.to(device)

    val_input = torch.tensor(pd.read_csv(
        data_folder + "/train_default/realistic_validation_input.csv", header=None).values, dtype=torch.float)
    val_input = val_input.to(device)
    val_guess_0 = torch.tensor(pd.read_csv(
        data_folder + "/train_default/realistic_validation_baseline_JS.csv", header=None).values, dtype=torch.float)
    val_guess_0 = val_guess_0.to(device)
    val_label = torch.tensor(pd.read_csv(
        data_folder + "/train_default/realistic_validation_label.csv", header=None).values, dtype=torch.float)
    val_label = val_label.to(device)
    return train_input, train_guess_0, train_label, val_input, val_guess_0, val_label