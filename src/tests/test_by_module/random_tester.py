import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from models.baseline import Baseline
from utilities.io import csv_to_tensor
from utilities.plotting import plot_all_metrics_vs_mutations,\
                               plot_metric_vs_mutations,\
                               plot_interval_metrics_vs_mutations,\
                               plot_interval_performance
from modules.combined_finetuner import CombinedFinetuner
from modules.combined_errorfinder import CombinedErrorfinder

# Load data
data_path = "../../../data/exp_final/test_random/"
inputs = csv_to_tensor(data_path + "test_random_input.csv", device='cpu')
baselines = csv_to_tensor(data_path + "test_random_baseline.csv", device='cpu')
labels = csv_to_tensor(data_path + "test_random_label.csv", device='cpu')
num_mut = labels[:, -1].unsqueeze(1)

# Make guess
models_path = "../../../trained_models/exp_final_2/"
finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "finetuner_random_low",
                              large_mum_mut_dir=models_path + "finetuner_random_large")
finetuner_guess = finetuner(mutation_dist=inputs,
                            baseline_guess=baselines,
                            num_mut=num_mut)

# Plot results
list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baselines, finetuner_guess]
# plot_all_metrics_vs_mutations( list_of_methods, list_of_guesses, labels, '')
plot_metric_vs_mutations(list_of_metrics=["accuracy %", "sens: tp/p %", "spec: tn/n %"],
                         list_of_methods=list_of_methods,
                         list_of_guesses=list_of_guesses,
                         label=labels,
                         plot_path='')

# Run errorfinder
errorfinder = CombinedErrorfinder(low_mum_mut_dir=models_path + "errorfinder_random_low",
                                  large_mum_mut_dir=models_path + "errorfinder_random_large")
upper_guess, lower_guess = errorfinder(finetuner_guess=finetuner_guess,
                                       num_mut=num_mut)

# Plot results
plot_interval_metrics_vs_mutations(labels, upper_guess, lower_guess, '')
plot_interval_performance(labels, upper_guess, lower_guess, list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')
