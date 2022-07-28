import os
import sys

import pandas as pd
import torch

from signet import DATA
from signet.modules.signet_module import SigNet
from signet.utilities.io import read_methods_guesses, csv_to_tensor
from signet.utilities.plotting import final_plot_all_metrics_vs_mutations, final_plot_interval_metrics_vs_mutations, plot_metric_vs_mutations_classifier, plot_time_vs_mutations

# Load data
data_path = "../../../data/exp_all/"
inputs = csv_to_tensor(data_path + "test_input.csv", device='cpu')
labels = csv_to_tensor(data_path + "test_label.csv", device='cpu')
num_mut = labels[:, -1].unsqueeze(1)

print("data loaded")

# Load model
path = "../../trained_models/"
signet = SigNet(classifier=path + "detector",
                finetuner_realistic_low=path + "finetuner_low",
                finetuner_realistic_large=path + "finetuner_large",
                errorfinder=path + "errorfinder",
                opportunities_name_or_path=None,
                signatures_path=DATA + "/data.xlsx",
                mutation_type_order=DATA + "/mutation_type_order.xlsx")

print("model read")

result = signet(inputs*labels[:, -1].reshape(-1, 1), numpy=False)

finetuner_guess, lower_bound, upper_bound, classification, normalized_input = result.convert_output(convert_to="tensor")
print(classification)
print("forwarded")

list_of_methods = ["decompTumor2Sig", "MutationalPatterns", "mutSignatures", "SignatureEstimationQP","YAPSA"]
list_of_guesses, label = read_methods_guesses('cpu', "exp_all", list_of_methods, data_folder="../../../data/")
list_of_methods += ['NNLS', 'Finetuner']
list_of_guesses += [signet.baseline_guess, finetuner_guess[:,:-1]]

final_plot_all_metrics_vs_mutations(list_of_methods=list_of_methods,
                                    list_of_guesses=list_of_guesses,
                                    label=labels,
                                    signatures=signet.signatures,
                                    mutation_distributions=inputs,
                                    folder_path="../../../plots/paper/")

classification_cutoff = 0.5
classification_results = (classification >= classification_cutoff).to(torch.int64)
plot_metric_vs_mutations_classifier(guess=classification_results,
                                    label=torch.ones((inputs.shape[0])).to(torch.int64),
                                    num_muts_list=labels[:, -1], 
                                    plot_path="../../../plots/paper/")

times = pd.read_csv('../../../data/exp_all/other_methods/times/all_times.csv', index_col=0)
num_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4, 1e5]
plot_time_vs_mutations(times, num_muts, plot_path="../../../plots/paper/", show=True)

final_plot_interval_metrics_vs_mutations(labels, upper_bound, lower_bound, list(pd.read_excel(DATA + "/data.xlsx").columns)[1:], plot_path="../../../plots/paper/", show=False)
