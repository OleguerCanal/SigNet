
import os
import sys

import numpy as np
import pandas as pd
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_signatures, read_real_data, tensor_to_csv
from utilities.data_generator import DataGenerator

if __name__ == "__main__":
    signatures = read_signatures("../data/data_v2.xlsx", mutation_type_order="../data/mutation_type_order.xlsx")
    data_generator = DataGenerator(signatures=signatures,
                                   seed=None,
                                   shuffle=True)
    
    # REALISTIC-LOOKING DATA
    experiment_id = "exp_v2"
    # real_input, _ = read_real_data(experiment_id="exp_real_data", device="cpu")
    # input_batch, label_batch = data_generator.make_similar_set(
    #     examples_input=real_input,
    #     n_augmentations=2)
    # tensor_to_csv(input_batch, "../data/%s/data_input.csv"%experiment_id)
    # tensor_to_csv(label_batch, "../data/%s/data_label.csv"%experiment_id)

    # RANDOM DATA
    # Train
    input_batch, label_batch = data_generator.make_random_set(
        "train", "low", normalize=True)
    tensor_to_csv(input_batch, "../data/%s/train_random_low_input.csv"%experiment_id)
    tensor_to_csv(label_batch, "../data/%s/train_random_low_label.csv"%experiment_id)

    input_batch, label_batch = data_generator.make_random_set(
        "train", "large", normalize=True)
    tensor_to_csv(input_batch, "../data/%s/train_random_large_input.csv"%experiment_id)
    tensor_to_csv(label_batch, "../data/%s/train_random_large_label.csv"%experiment_id)

    # Val
    input_batch, label_batch = data_generator.make_random_set(
    "val", "low", normalize=True)
    tensor_to_csv(input_batch, "../data/%s/val_random_low_input.csv"%experiment_id)
    tensor_to_csv(label_batch, "../data/%s/val_random_low_label.csv"%experiment_id)

    input_batch, label_batch = data_generator.make_random_set(
    "val", "large", normalize=True)
    tensor_to_csv(input_batch, "../data/%s/val_random_large_input.csv"%experiment_id)
    tensor_to_csv(label_batch, "../data/%s/val_random_large_label.csv"%experiment_id)

    # Test
    input_batch, label_batch = data_generator.make_random_set(
    "test", "low", normalize=True)
    tensor_to_csv(input_batch, "../data/%s/test_random/test_random_input.csv"%experiment_id)
    tensor_to_csv(label_batch, "../data/%s/test_random/test_random_label.csv"%experiment_id)
