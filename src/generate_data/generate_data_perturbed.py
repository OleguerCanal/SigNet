
import os
import sys

import numpy as np
import pandas as pd
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import csv_to_tensor, read_data, read_signatures, read_real_data, tensor_to_csv
from utilities.data_generator import DataGenerator


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python generate_data_perturbed v3"
    cosmic_version = str(sys.argv[1])

    if cosmic_version == 'v3':
        experiment_id = "exp_real_data"
        signatures = read_signatures("../data/data.xlsx", mutation_type_order="../data/mutation_type_order.xlsx")
    elif cosmic_version == 'v2':
        experiment_id = "exp_generator_v2"
        signatures = read_signatures("../data/data_v2.xlsx", mutation_type_order="../data/mutation_type_order.xlsx")
    else:
        print("Not implemented for this version of COSMIC.")

    data_generator = DataGenerator(signatures=signatures,
                                   seed=None,
                                   shuffle=True)
    
    # REALISTIC-LOOKING DATA
    # train_data, _ = read_data(device='cpu', 
    #                           experiment_id=experiment_id,
    #                           include_baseline=False,
    #                           source="generator_large")
    # realistic_weights = train_data.labels

    # REAL DATA:
    real_data = csv_to_tensor("../data/real_data/sigprofiler_normalized_PCAWG.csv",
                              device="cpu", header=0, index_col=0)
    real_data = real_data/torch.sum(real_data, axis=1).reshape(-1, 1)
    labels = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)                            
    realistic_weights = labels

    # Generate sets perturbing REAL DATA:
    # Large nummut
    train_input, train_label, val_input, val_label, test_input, test_label = data_generator.make_similar_set(examples_weight=realistic_weights,
                                                                                                             large_low="large",
                                                                                                             n_augmentations=10)
    tensor_to_csv(train_input, "../data/%s/train_perturbed_large_input.csv"%experiment_id)
    tensor_to_csv(train_label, "../data/%s/train_perturbed_large_label.csv"%experiment_id)
    tensor_to_csv(val_input, "../data/%s/val_perturbed_large_input.csv"%experiment_id)
    tensor_to_csv(val_label, "../data/%s/val_perturbed_large_label.csv"%experiment_id)
    tensor_to_csv(test_input, "../data/%s/test_perturbed_input.csv"%experiment_id)
    tensor_to_csv(test_label, "../data/%s/test_perturbed_label.csv"%experiment_id)

    # Low nummut
    train_input, train_label, val_input, val_label, test_input, test_label = data_generator.make_similar_set(examples_weight=realistic_weights,
                                                                                                             large_low="low",
                                                                                                             n_augmentations=10)
    tensor_to_csv(train_input, "../data/%s/train_perturbed_low_input.csv"%experiment_id)
    tensor_to_csv(train_label, "../data/%s/train_perturbed_low_label.csv"%experiment_id)
    tensor_to_csv(val_input, "../data/%s/val_perturbed_low_input.csv"%experiment_id)
    tensor_to_csv(val_label, "../data/%s/val_perturbed_low_label.csv"%experiment_id)

    # Generate sets with only REAL DATA:
    # Large nummut
    train_input, train_label, val_input, val_label, test_input, test_label = data_generator.make_real_set(large_low="large",
                                                                                                            repetitions=2,
                                                                                                            normalize=True)
    tensor_to_csv(train_input, "../data/%s/train_real_large_input.csv"%experiment_id)
    tensor_to_csv(train_label, "../data/%s/train_real_large_label.csv"%experiment_id)
    tensor_to_csv(val_input, "../data/%s/val_real_large_input.csv"%experiment_id)
    tensor_to_csv(val_label, "../data/%s/val_real_large_label.csv"%experiment_id)
    tensor_to_csv(test_input, "../data/%s/test_real_input.csv"%experiment_id)
    tensor_to_csv(test_label, "../data/%s/test_real_label.csv"%experiment_id)

    # Low nummut
    train_input, train_label, val_input, val_label, test_input, test_label = data_generator.make_real_set(large_low="low",
                                                                                                            repetitions=2,
                                                                                                            normalize=True)
    tensor_to_csv(train_input, "../data/%s/train_real_low_input.csv"%experiment_id)
    tensor_to_csv(train_label, "../data/%s/train_real_low_label.csv"%experiment_id)
    tensor_to_csv(val_input, "../data/%s/val_real_low_input.csv"%experiment_id)
    tensor_to_csv(val_label, "../data/%s/val_real_low_label.csv"%experiment_id)

    # Agument REAL DATA:
    augmented_labels = data_generator.augment_real_set()
    tensor_to_csv(augmented_labels, "../data/%s/augmented_real_data_labels.csv"%experiment_id)

    # Generate sets augmenting REAL DATA:
    # Large nummut
    train_input, train_label, val_input, val_label, test_input, test_label = data_generator.make_real_set_augmented(large_low="large",
                                                                                                                    repetitions=2,
                                                                                                                    normalize=True)
    tensor_to_csv(train_input, "../data/%s/train_augmented_real_large_input.csv"%experiment_id)
    tensor_to_csv(train_label, "../data/%s/train_augmented_real_large_label.csv"%experiment_id)
    tensor_to_csv(val_input, "../data/%s/val_augmented_real_large_input.csv"%experiment_id)
    tensor_to_csv(val_label, "../data/%s/val_augmented_real_large_label.csv"%experiment_id)
    tensor_to_csv(test_input, "../data/%s/test_augmented_real_input.csv"%experiment_id)
    tensor_to_csv(test_label, "../data/%s/test_augmented_real_label.csv"%experiment_id)

    # Low nummut
    train_input, train_label, val_input, val_label, test_input, test_label = data_generator.make_real_set_augmented(large_low="low",
                                                                                                                    repetitions=2,
                                                                                                                    normalize=True)
    tensor_to_csv(train_input, "../data/%s/train_augmented_real_low_input.csv"%experiment_id)
    tensor_to_csv(train_label, "../data/%s/train_augmented_real_low_label.csv"%experiment_id)
    tensor_to_csv(val_input, "../data/%s/val_augmented_real_low_input.csv"%experiment_id)
    tensor_to_csv(val_label, "../data/%s/val_augmented_real_low_label.csv"%experiment_id)