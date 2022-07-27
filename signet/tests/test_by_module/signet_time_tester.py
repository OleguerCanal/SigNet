import os
import sys
import time
import numpy as np

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.signet_module import SigNet
from utilities.io import read_methods_guesses, read_signatures, read_test_data, csv_to_tensor, write_final_outputs
from utilities.plotting import final_plot_all_metrics_vs_mutations, final_plot_interval_metrics_vs_mutations, plot_all_metrics_vs_mutations, plot_interval_metrics_vs_mutations, plot_interval_metrics_vs_sigs, plot_interval_performance, plot_metric_vs_mutations, plot_metric_vs_sigs, plot_metric_vs_mutations_classifier, plot_reconstruction, plot_weights
from utilities.metrics import get_reconstruction_error

# Read data
data_folder = "../../data/"

# Load data
data_path = "../../../data/exp_all/"
inputs = csv_to_tensor(data_path + "test_input.csv", device='cpu')
labels = csv_to_tensor(data_path + "test_label.csv", device='cpu')
num_mut = labels[:, -1].unique()
print("data loaded")

list_of_inputs = []
for mut in num_mut:
    list_of_inputs.append(inputs[labels[:, -1] == mut, :])

replicates = 5
times = np.zeros((replicates, len(num_mut)))
for i,input in enumerate(list_of_inputs):
    for k in range(replicates):
        # Load model
        path = "../../trained_models/"

        st = time.time()

        signet = SigNet(classifier=path + "detector",
                        finetuner_realistic_low=path + "finetuner_low",
                        finetuner_realistic_large=path + "finetuner_large",
                        errorfinder=path + "errorfinder",
                        opportunities_name_or_path=None,
                        signatures_path=data_folder + "data.xlsx",
                        mutation_type_order=data_folder + "mutation_type_order.xlsx")
        
        et = time.time()
        times[k, i] = et-st

        result = signet(inputs, numpy=False)

times_df = pd.DataFrame(times)
times_df.columns = num_mut
times_df.loc['mean'] = times_df.mean()
times_df.to_csv('SigNet_times.txt')
