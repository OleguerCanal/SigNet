
import os
import sys

import numpy as np
import pandas as pd
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_data, read_signatures, read_real_data, tensor_to_csv
from utilities.data_generator import DataGenerator


if __name__ == "__main__":

    signatures = read_signatures("../data/data.xlsx", mutation_type_order="../data/mutation_type_order.xlsx")
    data_generator = DataGenerator(signatures=signatures,
                                   seed=None,
                                   shuffle=True)
    
    # REALISTIC-LOOKING DATA
    experiment_id = "exp_final"
    train_data, _ = read_data(device='cpu', 
                              experiment_id=experiment_id, 
                              source="realistic_large")

    realistic_weights = train_data.labels

    train_size = int(5e5)
    val_size = int(5e3)
    test_size = int(1e3)
    # Low nummut
    input_batch, label_batch = data_generator.make_similar_set(examples_weight=realistic_weights,
                                                               large_or_low="low",
                                                               is_test=False,
                                                               n_augmentations=6)
    n_points = input_batch.size(0)
    train_input, train_labels = input_batch[:int(TRAIN_PROP*n_points), ...], label_batch[:int(TRAIN_PROP*n_points), ...]
    val_input, val_labels = input_batch[int(TRAIN_PROP*n_points):, ...], label_batch[int(TRAIN_PROP*n_points):, ...]
    tensor_to_csv(train_input[:train_size,:], "../data/%s/train_augmented_low_input.csv"%experiment_id)
    tensor_to_csv(train_labels[:train_size,:], "../data/%s/train_augmented_low_label.csv"%experiment_id)
    tensor_to_csv(val_input[:val_size,:], "../data/%s/val_augmented_low_input.csv"%experiment_id)
    tensor_to_csv(val_labels[:val_size,:], "../data/%s/val_augmented_low_label.csv"%experiment_id)

    # Large nummut
    input_batch, label_batch = data_generator.make_similar_set(examples_weight=realistic_weights,
                                                               large_or_low="large",
                                                               is_test=False,
                                                               n_augmentations=6)
    n_points = input_batch.size(0)
    train_input, train_labels = input_batch[:int(TRAIN_PROP*n_points), ...], label_batch[:int(TRAIN_PROP*n_points), ...]
    val_input, val_labels = input_batch[int(TRAIN_PROP*n_points):, ...], label_batch[int(TRAIN_PROP*n_points):, ...]
    tensor_to_csv(train_input[:train_size,:], "../data/%s/train_augmented_large_input.csv"%experiment_id)
    tensor_to_csv(train_labels[:train_size,:], "../data/%s/train_augmented_large_label.csv"%experiment_id)
    tensor_to_csv(val_input[:val_size,:], "../data/%s/val_augmented_large_input.csv"%experiment_id)
    tensor_to_csv(val_labels[:val_size,:], "../data/%s/val_augmented_large_label.csv"%experiment_id)

    # Test
    input_batch, label_batch = data_generator.make_similar_set(examples_weight=realistic_weights,
                                                                large_or_low=None,
                                                                is_test=True,
                                                                n_augmentations=2)
    tensor_to_csv(input_batch[:test_size,:], "../data/%s/test_augmented/test_augmented_input.csv"%experiment_id)
    tensor_to_csv(label_batch[:test_size,:], "../data/%s/test_augmented/test_augmented_label.csv"%experiment_id)


    # OLD RANDOM DATA
    # # Train
    # input_batch, label_batch = data_generator.make_random_set("train", "low", normalize=True)
    # tensor_to_csv(input_batch, "../data/%s/train_random_low_input.csv"%experiment_id)
    # tensor_to_csv(label_batch, "../data/%s/train_random_low_label.csv"%experiment_id)

    # input_batch, label_batch = data_generator.make_random_set("train", "large", normalize=True)
    # tensor_to_csv(input_batch, "../data/%s/train_random_large_input.csv"%experiment_id)
    # tensor_to_csv(label_batch, "../data/%s/train_random_large_label.csv"%experiment_id)

    # # Val
    # input_batch, label_batch = data_generator.make_random_set("val", "low", normalize=True)
    # tensor_to_csv(input_batch, "../data/%s/val_random_low_input.csv"%experiment_id)
    # tensor_to_csv(label_batch, "../data/%s/val_random_low_label.csv"%experiment_id)

    # input_batch, label_batch = data_generator.make_random_set("val", "large", normalize=True)
    # tensor_to_csv(input_batch, "../data/%s/val_random_large_input.csv"%experiment_id)
    # tensor_to_csv(label_batch, "../data/%s/val_random_large_label.csv"%experiment_id)

    # # Test
    # input_batch, label_batch = data_generator.make_random_set("test", "low", normalize=True)
    # tensor_to_csv(input_batch, "../data/%s/test_random/test_random_input.csv"%experiment_id)
    # tensor_to_csv(label_batch, "../data/%s/test_random/test_random_label.csv"%experiment_id)


