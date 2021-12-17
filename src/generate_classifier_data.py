import numpy as np
import torch

from utilities.io import read_signatures, tensor_to_csv, csv_to_tensor, read_model, read_data, read_test_data
from utilities.data_partitions import DataPartitions
from modules.combined_finetuner import CombinedFinetuner
from tqdm import tqdm


def shuffle(inputs, labels, num_mut):
    indexes = torch.randperm(inputs.shape[0])
    return inputs[indexes, ...], labels[indexes, ...], num_mut[indexes, ...]

if __name__ == "__main__":
    data_folder = "../data"
    signatures = read_signatures(data_folder + "/data.xlsx", mutation_type_order=data_folder+"/mutation_type_order.xlsx")
    experiment_id = "exp_good"

    # Read all realistic data    
    train_realistic_inputs = csv_to_tensor(data_folder + '/' + experiment_id + "/train_realistic_low_input.csv")
    train_realistic_nummut = csv_to_tensor(data_folder + '/' + experiment_id + "/train_realistic_low_label.csv")[:, -1].view(-1, 1)
    val_realistic_inputs = csv_to_tensor(data_folder + '/' + experiment_id + "/val_realistic_low_input.csv")
    val_realistic_nummut = csv_to_tensor(data_folder + '/' + experiment_id + "/val_realistic_low_label.csv")[:, -1].view(-1, 1)
    test_realistic_inputs = csv_to_tensor(data_folder + '/' + experiment_id + "/test_realistic/test_realistic_input.csv")
    test_realistic_nummut = csv_to_tensor(data_folder + '/' + experiment_id + "/test_realistic/test_realistic_label.csv")[:, -1].view(-1, 1)
    
    # Label all realistic data as a 1
    train_realistic_labels = torch.ones((train_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    val_realistic_labels = torch.ones((val_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    test_realistic_labels = torch.ones((test_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)

    # Read random data
    train_random, val_random = read_data(device="cpu",
                                         experiment_id=experiment_id,
                                         source="perturbed_low",
                                         data_folder=data_folder)
    test_random_inputs_ = csv_to_tensor(data_folder + '/' + experiment_id + "/test_perturbed/test_perturbed_input.csv")
    test_random_labels_ = csv_to_tensor(data_folder + '/' + experiment_id + "/test_perturbed/test_perturbed_label.csv")
    test_random_baseline_ = csv_to_tensor(data_folder + '/' + experiment_id + "/test_perturbed/test_perturbed_baseline.csv")
    test_random = DataPartitions(inputs=test_random_inputs_,
                                 labels=test_random_labels_,
                                 prev_guess=test_random_baseline_)

    # Label all random data as a 0
    train_random_labels = torch.zeros((train_random.inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    val_random_labels = torch.zeros((val_random.inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    test_random_labels = torch.zeros((test_random.inputs.shape[0], 1)).to(torch.float).view(-1, 1)


    train_classification_inputs = torch.cat((train_realistic_inputs, train_random.inputs))
    train_classification_labels = torch.cat((train_realistic_labels, train_random_labels))
    train_classification_numut = torch.cat((train_realistic_nummut, train_random.num_mut))
    train_classification_inputs, train_classification_labels, train_classification_numut =\
        shuffle(inputs=train_classification_inputs,
                labels=train_classification_labels,
                num_mut=train_classification_numut)

    val_classification_inputs = torch.cat((val_realistic_inputs, val_random.inputs))
    val_classification_labels = torch.cat((val_realistic_labels, val_random_labels))
    val_classification_numut = torch.cat((val_realistic_nummut, val_random.num_mut))
    val_classification_inputs, val_classification_labels, val_classification_numut =\
        shuffle(inputs=val_classification_inputs,
                labels=val_classification_labels,
                num_mut=val_classification_numut)
    
    # split val-train
    # val_classification_inputs = train_classification_inputs[-10000:, ...]
    # val_classification_labels = train_classification_labels[-10000:, ...]
    # val_classification_numut = train_classification_numut[-10000:, ...]

    # train_classification_inputs = train_classification_inputs[:-10000, ...]
    # train_classification_labels = train_classification_labels[:-10000, ...]
    # train_classification_numut = train_classification_numut[:-10000, ...]
    
    test_classification_inputs = torch.cat((test_realistic_inputs, test_random.inputs))
    test_classification_labels = torch.cat((test_realistic_labels, test_random_labels))
    test_classification_numut = torch.cat((test_realistic_nummut, test_random.num_mut))
    test_classification_inputs, test_classification_labels, test_classification_numut =\
         shuffle(inputs=test_classification_inputs,
                 labels=test_classification_labels,
                 num_mut=test_classification_numut)

    # store everything
    tensor_to_csv(train_classification_inputs, data_folder + '/' + experiment_id + "/classifier/train_input.csv")
    tensor_to_csv(train_classification_labels, data_folder + '/' + experiment_id + "/classifier/train_label.csv")
    tensor_to_csv(train_classification_numut, data_folder + '/' + experiment_id + "/classifier/train_num_mut.csv")

    tensor_to_csv(val_classification_inputs, data_folder + '/' + experiment_id + "/classifier/val_input.csv")
    tensor_to_csv(val_classification_labels, data_folder + '/' + experiment_id + "/classifier/val_label.csv")
    tensor_to_csv(val_classification_numut, data_folder + '/' + experiment_id + "/classifier/val_num_mut.csv")

    tensor_to_csv(test_classification_inputs, data_folder + '/' + experiment_id + "/classifier/test_input.csv")
    tensor_to_csv(test_classification_labels, data_folder + '/' + experiment_id + "/classifier/test_label.csv")
    tensor_to_csv(test_classification_numut, data_folder + '/' + experiment_id + "/classifier/test_num_mut.csv")


