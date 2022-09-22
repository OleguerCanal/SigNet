import os
import sys

import pandas as pd
import torch

from signet import DATA, TRAINED_MODELS
from signet.modules.signet_module import SigNet
from signet.utilities.io import read_methods_guesses
from signet.utilities.plotting import (final_plot_all_metrics_vs_mutations,
                                       final_plot_interval_metrics_vs_mutations,
                                       plot_metric_vs_mutations_classifier,
                                       plot_time_vs_mutations,
                                       final_plot_distance_vs_mutations,
                                       final_plot_intlen_metrics_vs_mutations)

# Load data NOTE! I'M NOT SURE IF THIS IS THE REAL DATA!!!!
data_path = DATA + "/datasets/"
inputs = pd.read_csv(data_path + "test_input.csv", header=None, index_col=None)
labels = pd.read_csv(data_path + "test_label.csv", header=None, index_col=None)
num_mut = labels[72]

print("data loaded")

signatures = pd.read_excel(DATA + '/data.xlsx')

input_df = inputs.mul(num_mut,0)
input_df.index = ['sample_' + str(i) for i in list(range(len(input_df.index)))]
input_df.columns = signatures['Type']

# Load model
signet = SigNet(classifier=TRAINED_MODELS + "detector",
                finetuner_realistic_low=TRAINED_MODELS + "finetuner_low",
                finetuner_realistic_large=TRAINED_MODELS + "finetuner_large",
                errorfinder=TRAINED_MODELS + "errorfinder",
                opportunities_name_or_path=None,
                signatures_path=DATA + "/data.xlsx",
                mutation_type_order=DATA + "/mutation_type_order.xlsx")

print("model read")
result = signet(input_df, numpy=False)
print("forwarded")

finetuner_guess, lower_bound, upper_bound, classification, normalized_input = result.get_output(format="tensor")
print("finetuner_guess", finetuner_guess, finetuner_guess.shape)

# list_of_methods = ["decompTumor2Sig", "MutationalPatterns", "mutSignatures", "SignatureEstimationQP","YAPSA"]
# list_of_guesses, label = read_methods_guesses('cpu', "exp_all", list_of_methods, data_folder="../../../data/")
# list_of_methods += ['NNLS', 'Finetuner']
# list_of_guesses += [signet.baseline_guess, finetuner_guess[:,:-1]]

# labels = torch.tensor(labels.values, dtype=torch.float)
# final_plot_all_metrics_vs_mutations(list_of_methods=list_of_methods,
#                                     list_of_guesses=list_of_guesses,
#                                     label=labels,
#                                     signatures=signet.signatures,
#                                     mutation_distributions=torch.tensor(inputs.values, dtype=torch.float),
#                                     folder_path="../../../plots/paper/")

labels = torch.tensor(labels.values)
# num_muts_list = torch.tensor(labels.values[:, -1])
# print(num_muts_list)

# classification_cutoff = 0.5
# classification_results = (classification >= classification_cutoff).to(torch.int64)
# plot_metric_vs_mutations_classifier(guess=classification_results,
#                                     label=torch.ones((inputs.shape[0])).to(torch.int64),
#                                     num_muts_list=num_muts_list, 
#                                     plot_path="../../../plots/paper/")

times = pd.read_csv(DATA + '/results/all_methods_times_norm.csv', index_col=0, header=None)
num_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4, 1e5]
plot_time_vs_mutations(times, 
                       num_muts, 
                       plot_path="../../plots/paper/times.pdf", 
                       show=False)

sigsnames = list(pd.read_excel(DATA + "/data.xlsx").columns)[1:]


final_plot_interval_metrics_vs_mutations(labels,
                                         upper_bound,
                                         lower_bound,
                                         sigsnames,
                                         plot_path="../../plots/paper/prop_out.pdf",
                                         show=False)
final_plot_distance_vs_mutations(labels,
                                 finetuner_guess,
                                 sigsnames,
                                 plot_path="../../plots/paper/guess_distance.pdf",
                                 show=False)

final_plot_intlen_metrics_vs_mutations(labels,
                                       upper_bound,
                                       lower_bound,
                                       sigsnames,
                                       plot_path="../../plots/paper/interval_length.pdf",
                                       show=False)