import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.signet import SigNet
from utilities.io import read_test_data, csv_to_tensor
from utilities.plotting import plot_all_metrics_vs_mutations, plot_interval_metrics_vs_mutations, plot_interval_metrics_vs_sigs, plot_interval_performance, plot_metric_vs_mutations, plot_metric_vs_sigs, plot_metric_vs_mutations_classifier

# Read data
data_folder = "../../../data/"
inputs, label = read_test_data(device='cpu',
                               experiment_id="exp_final",
                               test_id="test",
                               data_folder=data_folder)

# data_path = data_folder + "exp_final/test_realistic/"
# inputs = csv_to_tensor(data_path + "test_realistic_input.csv", device='cpu')
# label = csv_to_tensor(data_path + "test_realistic_label.csv", device='cpu')

# data_path = data_folder + "exp_final/test_random/"
# inputs = csv_to_tensor(data_path + "test_random_input.csv", device='cpu')
# label = csv_to_tensor(data_path + "test_random_label.csv", device='cpu')

print("data loaded")

# Load model
path = "../../../trained_models/exp_final_2/"
signet = SigNet(classifier=path + "classifier",
                finetuner_random_low=path + "finetuner_random_low",
                finetuner_random_large=path + "finetuner_random_large",
                finetuner_realistic_low=path + "finetuner_realistic_low",
                finetuner_realistic_large=path + "finetuner_realistic_large",
                errorfinder=path + "errorfinder_realistic_low",
                opportunities_name_or_path=None,
                signatures_path=data_folder + "data.xlsx",
                mutation_type_order=data_folder + "mutation_type_order.xlsx")

print("model read")

finetuner_guess, upper_bound, lower_bound = signet(inputs*label[:, -1].reshape(-1, 1), numpy=False)

print("forwarded")
# plot_all_metrics_vs_mutations(list_of_methods=['Baseline', 'Finetuner'],
#                               list_of_guesses=[signet.baseline_guess, finetuner_guess],
#                               label=label,
#                               show=True)

# plot_metric_vs_mutations_classifier(guess=signet.finetuner_errorfinder.classification_results,
#                                     label=torch.ones((inputs.shape[0])),
#                                     num_muts_list=label[:, -1])

plot_metric_vs_mutations(list_of_metrics=["accuracy %", "sens: tp/p %", "spec: tn/n %"],
                         list_of_methods=['Baseline', 'Finetuner'],
                         list_of_guesses=[signet.baseline_guess, finetuner_guess],
                         label=label,
                         show=True)

plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, show=True)
plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel(data_folder + "data.xlsx").columns)[1:], show=True)
# plot_interval_metrics_vs_sigs(label, upper_bound, lower_bound, '')

# errorfiner_realistic = read_model(model_path + experiment_id + "/errorfinder_realistic")
# upper_bound, lower_bound = errorfiner_realistic(finetuner_guess, label[:,-1].reshape(-1,1))

# plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, '')
# plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')