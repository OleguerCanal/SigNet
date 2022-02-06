import sys

import torch
import matplotlib.pyplot as plt
import numpy as np

from utilities.data_generator import DataGenerator
from utilities.io import read_model, read_data_generator, read_signatures, tensor_to_csv



if __name__ == "__main__":

    cosmic_version = str(sys.argv[1])

    if cosmic_version == 'v3':
        generator_model_path = "../trained_models/exp_generator/generator"
        experiment_id = "exp_generator"
        signatures = read_signatures("../data/data.xlsx", mutation_type_order="../data/mutation_type_order.xlsx")
    elif cosmic_version == 'v2':
        generator_model_path = "../trained_models/exp_generator_v2/generator"
        experiment_id = "exp_generator_v2"
        signatures = read_signatures("../data/data_v2.xlsx", mutation_type_order="../data/mutation_type_order.xlsx")
    else:
        raise NotImplementedError

    data_generator = DataGenerator(signatures=signatures,
                                   seed=None,
                                   shuffle=True)

    ## Low nummut
    train_input, train_labels = data_generator.make_realistic_set(generator_model_path=generator_model_path,
                                                                 large_low="low",
                                                                 set="train")
    print("done")
    tensor_to_csv(train_input, "../data/%s/train_generator_low_input.csv"%experiment_id)
    print("saved first")
    tensor_to_csv(train_labels, "../data/%s/train_generator_low_label.csv"%experiment_id)
    print("saved second")

    val_input, val_labels = data_generator.make_realistic_set(generator_model_path=generator_model_path,
                                                                 large_low="low",
                                                                 set="val")

    tensor_to_csv(val_input, "../data/%s/val_generator_low_input.csv"%experiment_id)
    tensor_to_csv(val_labels, "../data/%s/val_generator_low_label.csv"%experiment_id)

    ## Large nummut
    train_input, train_labels = data_generator.make_realistic_set(generator_model_path=generator_model_path,
                                                                 large_low="large",
                                                                 set="train")

    tensor_to_csv(train_input, "../data/%s/train_generator_large_input.csv"%experiment_id)
    tensor_to_csv(train_labels, "../data/%s/train_generator_large_label.csv"%experiment_id)

    val_input, val_labels = data_generator.make_realistic_set(generator_model_path=generator_model_path,
                                                                 large_low="large",
                                                                 set="val")

    tensor_to_csv(val_input, "../data/%s/val_generator_large_input.csv"%experiment_id)
    tensor_to_csv(val_labels, "../data/%s/val_generator_large_label.csv"%experiment_id)

    ## Test
    test_input, test_labels = data_generator.make_realistic_set(generator_model_path=generator_model_path,
                                                                 large_low="large",
                                                                 set="test")

    tensor_to_csv(test_input, "../data/%s/test_generator_input.csv"%experiment_id)
    tensor_to_csv(test_labels, "../data/%s/test_generator_label.csv"%experiment_id)
