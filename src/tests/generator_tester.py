import os
import sys

import pandas as pd
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.plotting import plot_metric_vs_mutations, plot_metric_vs_sigs
from utilities.io import read_signatures, read_test_data
from models.generator import Generator
from models.yapsa_inspired_baseline import YapsaInspiredBaseline
from models.finetuner import FineTuner

test_id = "test_random"
device = "cpu"

experiment_id = "exp_0"
# Model params finetuner
model_id_finetuner = "finetuner_random"
num_hidden_layers = 3
num_neurons = 600
num_classes = 72

# Open data
signatures = read_signatures("../../data/data.xlsx", num_classes=72)
input_batch, label_mut_batch = read_test_data(device=device,
                                              experiment_id=experiment_id,
                                              test_id=test_id,
                                              data_folder="../../data")
n_datapoints = -1  # What is this for?
input_batch = input_batch[:n_datapoints]
label_batch = label_mut_batch[:n_datapoints, :-1]
num_mut = label_mut_batch[:n_datapoints, -1].reshape((-1, 1))
label_mut_batch = label_mut_batch[:n_datapoints]

# Baseline:
baseline = YapsaInspiredBaseline(signatures)
baseline_batch = baseline.get_weights_batch(input_batch)  # [:50, ...])

# Instantiate model and do predictions for finetuner:
finetuner_model = FineTuner(num_classes=num_classes,
                            num_hidden_layers=num_hidden_layers,
                            num_units=num_neurons)
finetuner_model.load_state_dict(torch.load(os.path.join(
    "../../trained_models/" + experiment_id, model_id_finetuner), map_location=torch.device(device)))
finetuner_model.eval()
finetuner_guessed_labels = finetuner_model(
    input_batch, baseline_batch, num_mut)


experiment_id = "exp_real_data"
# Model params generator
model_id_finetuner = "generator_real"
num_hidden_layers = 3
num_neurons = 600

# Instantiate model and do predictions for generator:
generator_finetuner = FineTuner(num_classes=num_classes,
                                num_hidden_layers=num_hidden_layers,
                                num_units=num_neurons)
generator_model = Generator(baseline=baseline,
                            finetuner=generator_finetuner,
                            signatures=signatures)
generator_model.load_state_dict(torch.load(os.path.join(
    "../../trained_models/" + experiment_id, model_id_finetuner), map_location=torch.device(device)))
generator_model.eval()
generator_guessed_labels = generator_model.get_finetuned_weights(mutation_dist=input_batch,
                                                                 num_mut=num_mut)


# Check performance
list_of_guesses = [finetuner_guessed_labels, generator_guessed_labels]
list_of_methods = ["Finetuner", "Generator"]
list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses,
                         label_mut_batch, "../../plots/exp_real_data/random_vs_num_muts")
plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses,
                    label_mut_batch, "../../plots/exp_real_data/random_vs_sigs")

list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses,
                    label_mut_batch, "../../plots/exp_real_data/random_performance_vs_sigs")
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses,
                         label_mut_batch, "../../plots/exp_real_data/random_performance_vs_num_muts")
