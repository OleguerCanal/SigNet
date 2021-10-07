import os
import pathlib

import json
import pandas as pd
import torch

from utilities.data_partitions import DataPartitions
from models.classifier import Classifier
from models.finetuner import FineTuner
from models.error_finder import ErrorFinder

def read_signatures(file, num_classes=72):
    signatures_data = pd.read_excel(file)
    signatures = [torch.tensor(signatures_data.iloc[:, i]).type(torch.float32)
                  for i in range(2, num_classes + 2)][:num_classes]
    signatures = torch.stack(signatures).t()
    return signatures


def csv_to_tensor(file, device="cpu"):
    input_tensor = torch.tensor(pd.read_csv(
        file, header=None).values, dtype=torch.float)
    assert(not torch.isnan(input_tensor).any())
    # assert(torch.count_nonzero(torch.sum(input_tensor, axis=1))
    #        == input_tensor.shape[0])
    return input_tensor.float().to(device)

def tensor_to_csv(data_tensor, output_path):
    directory = os.path.dirname(output_path)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    df = data_tensor.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv(output_path, header=False, index=False) 

def read_data(device, experiment_id, source, data_folder="../data"):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    # assert(source in ["random", "realistic", "mixed"])
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

def read_data_classifier(device, experiment_id, data_folder="../data"):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    path = os.path.join(data_folder, experiment_id)

    train_input = csv_to_tensor(path + "/train_input.csv", device)
    train_num_mut = csv_to_tensor(path + "/train_num_mut.csv", device).reshape((-1,1))
    train_label = csv_to_tensor(path + "/train_label.csv", device).reshape((-1,1))

    train_data = DataPartitions(inputs=train_input,
                                num_mut=train_num_mut,
                                labels=train_label)

    val_input = csv_to_tensor(path + "/val_input.csv", device)
    val_num_mut = csv_to_tensor(path + "/val_num_mut.csv", device).reshape((-1,1))
    val_label = csv_to_tensor(path + "/val_label.csv", device).reshape((-1,1))

    val_data = DataPartitions(inputs=val_input,
                              num_mut=val_num_mut,
                              labels=val_label)

    return train_data, val_data

def read_real_data(device, experiment_id, data_folder="../data"):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    path = os.path.join(data_folder, experiment_id)

    real_input = csv_to_tensor(path + "/real_data_input.csv", device)
    real_num_mut = csv_to_tensor(path + "/real_data_num_mut.csv", device)

    return real_input, real_num_mut

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


def read_model(directory):
    """Instantiate a pre-trained model from the stored vars
    The model is in cpu and in eval mode

    Args:
        directory (String): Folder containing state_dict and init_args.json of the model
    """
    # Load init_args
    init_args_file = os.path.join(directory, 'init_args.json')
    with open(init_args_file, 'r') as fp:
        init_args = json.load(fp)
    model_type = init_args["model_type"]
    init_args.pop("model_type")
    assert(model_type is not None)  # Model type not saved!
    assert(model_type in ["Classifier", "FineTuner", "ErrorFinder"])

    # Instantiate model class
    if model_type == "Classifier":
        model = Classifier(**init_args)
    elif model_type == "FineTuner":
        model = FineTuner(**init_args)
    elif model_type == "ErrorFinder":
        model = ErrorFinder(**init_args)
    
    # Load model weights
    state_dict_file = os.path.join(directory, "state_dict")
    try:
        state_dict = torch.load(f=state_dict_file,
                                map_location=torch.device('cpu'))
    except:
        state_dict = torch.load(f=state_dict_file + ".zip",
                                map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def save_model(model, directory):
    """Store a pytorch model. The arguments are splitted into 2 files:
    A init_args.json needed to instantiate the class, and the model state_dict

    Args:
        model (nn.Module): Model to save
        directory (String): Path where to save
    """
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Store init_args into a json file
    init_args_file = os.path.join(directory, 'init_args.json')
    with open(init_args_file, 'w') as fp:
        json.dump(model.init_args, fp)

    # Store state_dict
    state_dict_file = os.path.join(directory, "state_dict")
    torch.save(model.state_dict(), state_dict_file)
