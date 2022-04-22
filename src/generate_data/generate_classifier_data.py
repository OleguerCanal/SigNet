import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_signatures, tensor_to_csv, csv_to_tensor
from utilities.data_generator import DataGenerator

def shuffle(inputs, labels, num_mut):
    indexes = torch.randperm(inputs.shape[0])
    return inputs[indexes, ...], labels[indexes, ...], num_mut[indexes, ...]

if __name__ == "__main__":

    assert len(sys.argv) == 2, "Usage: python generate_classifier_data.py v3"
    cosmic_version = str(sys.argv[1])

    if cosmic_version == 'v3':
        experiment_id = "exp_oversample"
        signatures = read_signatures("../../data/data.xlsx", mutation_type_order="../../data/mutation_type_order.xlsx")
    elif cosmic_version == 'v2':
        experiment_id = "exp_generator_v2"
        signatures = read_signatures("../../data/data_v2.xlsx", mutation_type_order="../../data/mutation_type_order.xlsx")
    else:
        raise NotImplementedError

    data_folder = "../../data"

    # Read all oversampled real data
    train_realistic_inputs = csv_to_tensor(data_folder + '/' + experiment_id + "/train_low_input.csv")
    train_realistic_nummut = csv_to_tensor(data_folder + '/' + experiment_id + "/train_low_label.csv")[:, -1].view(-1, 1)
    val_realistic_inputs = csv_to_tensor(data_folder + '/' + experiment_id + "/val_low_input.csv")
    val_realistic_nummut = csv_to_tensor(data_folder + '/' + experiment_id + "/val_low_label.csv")[:, -1].view(-1, 1)
    test_realistic_inputs = csv_to_tensor(data_folder + '/' + experiment_id + "/test_input.csv")
    test_realistic_nummut = csv_to_tensor(data_folder + '/' + experiment_id + "/test_label.csv")[:, -1].view(-1, 1)
    
    # Label all realistic data as a 1
    train_realistic_labels = torch.ones((train_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    val_realistic_labels = torch.ones((val_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)
    test_realistic_labels = torch.ones((test_realistic_inputs.shape[0], 1)).to(torch.float).view(-1, 1)

    # Read random data

    data_generator = DataGenerator(signatures=signatures,
                                    seed=None,
                                    shuffle=True)

    train_random_input, train_random_label = data_generator.make_random_set(set='train',
                                                              large_low='low',
                                                              num_samples=int(train_realistic_inputs.shape[0]/8),
                                                              min_n_signatures=1,
                                                              max_n_signatures=10,
                                                              normalize=True)
    val_random_input, val_random_label = data_generator.make_random_set(set='val',
                                                              large_low='low',
                                                              num_samples=int(val_realistic_inputs.shape[0]/8),
                                                              min_n_signatures=1,
                                                              max_n_signatures=10,
                                                              normalize=True)
    test_random_input, test_random_label = data_generator.make_random_set(set='test',
                                                              large_low='low',
                                                              num_samples=int(test_realistic_inputs.shape[0]/10),
                                                              min_n_signatures=1,
                                                              max_n_signatures=10,
                                                              normalize=True)


    # Label all random data as a 0
    train_random_labels = torch.zeros((train_random_input.shape[0], 1)).to(torch.float).view(-1, 1)
    val_random_labels = torch.zeros((val_random_input.shape[0], 1)).to(torch.float).view(-1, 1)
    test_random_labels = torch.zeros((test_random_input.shape[0], 1)).to(torch.float).view(-1, 1)


    train_classification_inputs = torch.cat((train_realistic_inputs, train_random_input))
    train_classification_labels = torch.cat((train_realistic_labels, train_random_labels))
    train_classification_numut = torch.cat((train_realistic_nummut, train_random_label[:,-1].view(-1, 1)))
    train_classification_inputs, train_classification_labels, train_classification_numut =\
        shuffle(inputs=train_classification_inputs,
                labels=train_classification_labels,
                num_mut=train_classification_numut)

    val_classification_inputs = torch.cat((val_realistic_inputs, val_random_input))
    val_classification_labels = torch.cat((val_realistic_labels, val_random_labels))
    val_classification_numut = torch.cat((val_realistic_nummut, val_random_label[:, -1].view(-1, 1)))
    val_classification_inputs, val_classification_labels, val_classification_numut =\
        shuffle(inputs=val_classification_inputs,
                labels=val_classification_labels,
                num_mut=val_classification_numut)
    
    test_classification_inputs = torch.cat((test_realistic_inputs, test_random_input))
    test_classification_labels = torch.cat((test_realistic_labels, test_random_labels))
    test_classification_numut = torch.cat((test_realistic_nummut, test_random_label[:, -1].view(-1, 1)))
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


