import os
import pathlib
from torch.utils import data
import yaml

import json
import pandas as pd
import torch

from utilities.data_partitions import DataPartitions
from utilities.generator_data import GeneratorData
from models.generator import Generator
from models.classifier import Classifier
from models.finetuner import FineTuner
from models.error_finder import ErrorFinder
from utilities.metrics import get_reconstruction_error

def read_signatures(file, mutation_type_order="../../data/mutation_type_order.xlsx"):
    """
    File must contain first column with mutations types X[Y>Z]W and the rest of the columns must be the set of signatures
    """
    # Sort according to cosmic mutation types order
    signatures_data = sort_signatures(file, mutation_type_order=mutation_type_order)

    num_sigs = len(signatures_data.columns) - 1
    signatures = [torch.tensor(signatures_data.iloc[:, i]).type(torch.float32)
                  for i in range(1, num_sigs + 1)][:num_sigs]
    signatures = torch.stack(signatures).t()
    return signatures

def sort_signatures(file, output_file=None, mutation_type_order="../../data/mutation_type_order.xlsx"):
    signatures_data = pd.read_excel(file)
    mutation_order = pd.read_excel(mutation_type_order)

    # Sort according to cosmic mutation types order
    signatures_data.rename(columns = {list(signatures_data)[0]:'Type'}, inplace=True)
    signatures_data = signatures_data.set_index('Type')
    signatures_data = signatures_data.reindex(index=mutation_order['Type'])
    signatures_data = signatures_data.reset_index()

    if output_file is not None:
        create_dir(output_file)
        signatures_data.to_csv(output_file, index=False)
    return signatures_data

def csv_to_tensor(file, device="cpu", header=None, index_col=None):
    input_tensor = torch.tensor(pd.read_csv(
        file, header=header, index_col=index_col).values, dtype=torch.float)
    assert(not torch.isnan(input_tensor).any())
    # assert(torch.count_nonzero(torch.sum(input_tensor, axis=1))
    #        == input_tensor.shape[0])
    return input_tensor.float().to(device)

def tensor_to_csv(data_tensor, output_path):
    create_dir(output_path)
    df = data_tensor.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv(output_path, header=False, index=False) 

def read_data(device, experiment_id, source, data_folder="../data", include_baseline=True, include_labels=True):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to "../data".
    """
    # assert(source in ["random", "realistic", "perturbed"])
    path = os.path.join(data_folder, experiment_id)

    train_input = csv_to_tensor(path + "/train_%s_input.csv" % source, device)
    train_baseline = csv_to_tensor(path + "/train_%s_baseline.csv" % source, device) if include_baseline else None
    train_label = csv_to_tensor(path + "/train_%s_label.csv" % source, device) if include_labels else None

    train_data = DataPartitions(inputs=train_input,
                                prev_guess=train_baseline,
                                labels=train_label)

    val_input = csv_to_tensor(path + "/val_%s_input.csv" % source, device)
    val_baseline = csv_to_tensor(path + "/val_%s_baseline.csv" % source, device) if include_baseline else None
    val_label = csv_to_tensor(path + "/val_%s_label.csv" % source, device) if include_labels else None

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

def read_data_generator(device, data_folder="../data"):

    real_data = csv_to_tensor(data_folder + "/real_data/sigprofiler_normalized_PCAWG.csv",
                              device=device, header=0, index_col=0)
    real_data = real_data/torch.sum(real_data, axis=1).reshape(-1, 1)
    real_data = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)
    data = real_data[torch.randperm(real_data.size()[0]),:]

    train_input = data[:int(real_data.size()[0]*0.95)]
    val_input = data[int(real_data.size()[0]*0.95):]

    train_data = GeneratorData(inputs=train_input)
    val_data = GeneratorData(inputs=val_input)

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
            path + "/other_methods/%s_guess.csv" % (method), device))

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


def read_model(directory, device="cpu"):
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
    assert(model_type in ["Classifier", "FineTuner", "ErrorFinder", "Generator"])
    if "device" in init_args.keys():
        init_args["device"] = device
        
    # Instantiate model class
    if model_type == "Generator":
        model = Generator(**init_args)
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
                                map_location=torch.device(device))
    except:
        state_dict = torch.load(f=state_dict_file + ".zip",
                                map_location=torch.device(device))
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


def update_dict(config, args):
    for arg in vars(args):
        if getattr(args, arg) is not None and arg in config:
            config[arg] = getattr(args, arg)[0]
    return config

def read_config(path):
    with open(path, 'r') as stream:
        data = yaml.safe_load(stream)
    return data["config"]

def create_dir(filepath):
    directory = os.path.dirname(filepath)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

def write_result(result, filepath):
    directory = os.path.dirname(filepath)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    fout = open(filepath, 'w')
    fout.write(str(result))
    fout.close()

def write_final_output(output_path, output_values, input_indexes, sigs_path="../../data/data.xlsx"):
    create_dir(output_path)
    sig_names = list(pd.read_excel(sigs_path).columns)[1:]
    df = pd.DataFrame(output_values)
    df.columns = sig_names
    df.index = input_indexes
    df.to_csv(output_path, header=True, index=True)

def write_final_outputs(weights, lower_bound, upper_bound, baseline, classification, reconstruction_error, input_file, output_path):
    create_dir(output_path + "/whatever.txt")
    sig_names = list(pd.read_excel("../../data/data.xlsx").columns)[1:]
    
    # Write results weight guesses
    df = pd.DataFrame(weights)
    df.columns = sig_names
    row_names =input_file.index.tolist()
    df.index = row_names
    df.to_csv(output_path + "/weight_guesses.csv", header=True, index=True)

    # Write results lower bound guesses
    df = pd.DataFrame(lower_bound)
    df.columns = sig_names
    row_names =input_file.index.tolist()
    df.index = row_names
    df.to_csv(output_path + "/lower_bound_guesses.csv", header=True, index=True)

    # Write results upper bound guesses
    df = pd.DataFrame(upper_bound)
    df.columns = sig_names
    row_names =input_file.index.tolist()
    df.index = row_names
    df.to_csv(output_path + "/upper_bound_guesses.csv", header=True, index=True)

    # Write results baseline guesses
    df = pd.DataFrame(baseline)
    df.columns = sig_names
    row_names =input_file.index.tolist()
    df.index = row_names
    df.to_csv(output_path + "/baseline_guesses.csv", header=True, index=True)

    # Write results classification
    df = pd.DataFrame(classification)
    df.columns = ["classification"]
    row_names =input_file.index.tolist()
    df.index = row_names
    df.to_csv(output_path + "/classification_guesses.csv", header=True, index=True)

    # Write results reconstruction error
    df = pd.DataFrame(reconstruction_error)
    df.columns = ["reconstruction_error"]
    row_names =input_file.index.tolist()
    df.index = row_names
    df.to_csv(output_path + "/reconstruction_error.csv", header=True, index=True)
