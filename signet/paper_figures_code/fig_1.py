import os
from pickle import TRUE
import sys

import pandas as pd
import torch

from signet import DATA, TRAINED_MODELS
from signet.modules.signet_module import SigNet
from signet.utilities.io import read_methods_guesses, tensor_to_csv
from signet.utilities.plotting import (final_plot_all_metrics_vs_mutations,
                                       final_plot_interval_metrics_vs_mutations, plot_distance_vs_mutations_all_methods,
                                       plot_metric_vs_mutations_classifier, plot_percentage_all_methods, plot_percentage_all_methods_fig1,
                                       plot_time_vs_mutations,
                                       final_plot_distance_vs_mutations,
                                       final_plot_intlen_metrics_vs_mutations,
                                       plot_violins_error)

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

tensor_to_csv(signet.baseline_guess, '../../data/exp_all/other_methods/all_results/NNLS_guess.csv')
tensor_to_csv(finetuner_guess[:,:-1], '../../data/exp_all/other_methods/all_results/Finetuner_guess.csv')
print('WRITING DONE!')

list_of_methods = ["decompTumor2Sig", "MutationalPatterns", "mutSignatures", "SignatureEstimationQP", "YAPSA",
                   "deconstructSigs", "mutationalCone", "QPsig", "sigLASSO", "MuSiCal", "NNLS", "SigNet"]
list_of_guesses, label = read_methods_guesses('cpu', "exp_all", list_of_methods, data_folder="../../data/")
list_of_methods = ['SigNet'] + list_of_methods
list_of_guesses = [finetuner_guess] + list_of_guesses
list_of_methods += ['NNLS', 'Finetuner']
list_of_guesses += [signet.baseline_guess, finetuner_guess[:,:-1]]

labels = torch.tensor(labels.values, dtype=torch.float)
# final_plot_all_metrics_vs_mutations(list_of_methods=list_of_methods,
#                                     list_of_guesses=list_of_guesses,
#                                     label=labels,
#                                     signatures=signet.signatures,
#                                     mutation_distributions=torch.tensor(inputs.values, dtype=torch.float),
#                                     folder_path="../../plots/paper/")

labels = torch.tensor(labels.values)
# num_muts_list = torch.tensor(labels.values[:, -1])
# print(num_muts_list)

# classification_cutoff = 0.5
# classification_results = (classification >= classification_cutoff).to(torch.int64)
# plot_metric_vs_mutations_classifier(guess=classification_results,
#                                     label=torch.ones((inputs.shape[0])).to(torch.int64),
#                                     num_muts_list=num_muts_list, 
#                                     plot_path="../../plots/paper/")

# times = pd.read_csv('../../data/exp_all/other_methods/all_results/all_methods_times_norm.csv', index_col=0, header=None)
# times = times.iloc[:,:-1]
# print(times)
# num_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4]
# plot_time_vs_mutations(times, 
#                        num_muts, 
#                        plot_path="../../plots/paper/times_MuSiCal.pdf", 
#                        show=False)

sigsnames = list(pd.read_excel(DATA + "/data.xlsx").columns)[1:]

# plot_percentage_all_methods(label, list_of_guesses, list_of_methods, sigsnames, plot_path=None, show=True, title=None)
# plot_percentage_all_methods_fig1(label, list_of_guesses, list_of_methods, sigsnames, plot_path=None, show=True, title=None)
plot_percentage_all_methods_fig1(label, list_of_guesses, list_of_methods, sigsnames, plot_path='SigNet_percentage_present_fig1.pdf', show=False, title=None)
# plot_percentage_all_methods(label, list_of_guesses, list_of_methods, sigsnames, plot_path='SigNet_supp_percentage_present.pdf', show=False, title=None)

# Select what signatures we should plot:
sigs_inds = list(range(29)) + [30] + list(range(32,46)) + [47,48,49] + [55,56,57,58,60,62,64]
sigs_names = [sigsnames[i] for i in sigs_inds]
label = label[:,sigs_inds+[-1]]
list_of_guesses = [finetuner_guess]
for i in range(len(list_of_guesses)):
  list_of_guesses[i] = list_of_guesses[i][:,sigs_inds]

plot_distance_vs_mutations_all_methods(label, list_of_guesses, list_of_methods,
                                        sigs_names, plot_path='../../plots/paper/error_all.pdf', show=False, title=None)
                                        # sigs_names, plot_path=None, show=True, title=None)
# Violins plots of the error
plot_violins_error(labels, finetuner_guess[:,:-1], sigsnames, plot_path=['violin.pdf','mean_violin.pdf'], show=False, title=None)

# Select what signatures we should plot:
# sigs_inds = list(range(29)) + [30] + list(range(32,46)) + [47,48,49] + [55,56,57,58,60,62,64]
# sigs_names = [sigsnames[i] for i in sigs_inds]
# label = labels[:,sigs_inds+[-1]]
# list_of_guesses = [finetuner_guess]
# for i in range(len(list_of_guesses)):
#   list_of_guesses[i] = list_of_guesses[i][:,sigs_inds]

# plot_distance_vs_mutations_all_methods(label, list_of_guesses, ['SigNet'],
#                                         sigs_names, plot_path='../../plots/paper/signet_error.pdf', show=False, title=None)
#                                         # sigs_names, plot_path=None, show=True, title=None)

plot_distance_vs_mutations_all_methods(label, list_of_guesses[5:], list_of_methods[5:],
                                        sigs_names, plot_path='../../plots/paper/error2.pdf', show=False, title=None)


# final_plot_interval_metrics_vs_mutations(labels,
#                                          upper_bound,
#                                          lower_bound,
#                                          sigsnames,
#                                          plot_path="../../plots/paper/prop_out.pdf",
#                                          show=False)
# labels2 = labels[labels[:,2]>0,:]
# finetuner_guess2 = finetuner_guess[labels[:,2]>0,:]

# finetuner_guess2 = finetuner_guess2[labels2[:,4]>0,:]
# labels2 = labels2[labels2[:,4]>0,:]

# finetuner_guess2 = finetuner_guess2[labels2[:,44]==0,:]
# labels2 = labels2[labels2[:,44]==0,:]

# finetuner_guess2 = torch.cat((finetuner_guess2[:,:-1], (finetuner_guess2[:,4] + finetuner_guess2[:,2]).view(-1,1), 
#                               # (finetuner_guess2[:,4] + finetuner_guess2[:,2]).view(-1,1), 
#                               # (finetuner_guess2[:,4] + finetuner_guess2[:,2]+finetuner_guess2[:,44]).view(-1,1), 
#                             finetuner_guess2[:,-1].view(-1,1)),1)
# labels2 = torch.cat((labels2[:,:-1], (labels2[:,4] + labels2[:,2]).view(-1,1),
#                             # (labels2[:,4] + labels2[:,2]).view(-1,1), 
#                             #   (labels2[:,4] + labels2[:,2] + labels2[:,44]).view(-1,1), 
#                             labels2[:,-1].view(-1,1)),1)
# sigsnames = sigsnames + ['SBS5+SBS40'] #, 'SBS3+SBS5', 'SBS3+SBS5+SBS40']
# final_plot_distance_vs_mutations(labels2,
#                                  finetuner_guess2,
#                                  sigsnames,
#                                  plot_path="../../plots/paper/guess_distance_5_40_restrict_labels.pdf",
#                                  show=True)

# final_plot_intlen_metrics_vs_mutations(labels,
#                                        upper_bound,
#                                        lower_bound,
#                                        sigsnames,
#                                        plot_path="../../plots/paper/interval_length.pdf",
#                                        show=False)

