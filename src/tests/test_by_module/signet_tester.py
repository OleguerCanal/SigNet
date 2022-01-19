import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.signet import SigNet
from utilities.io import read_signatures, read_test_data, csv_to_tensor, write_final_outputs
from utilities.plotting import plot_all_metrics_vs_mutations, plot_interval_metrics_vs_mutations, plot_interval_metrics_vs_sigs, plot_interval_performance, plot_metric_vs_mutations, plot_metric_vs_sigs, plot_metric_vs_mutations_classifier, plot_reconstruction, plot_weights
from utilities.metrics import get_reconstruction_error

# Read data
data_folder = "../../../data/"

# Load data
data_path = "../../../data/exp_generator/"
inputs = csv_to_tensor(data_path + "test_generator/test_generator_input.csv", device='cpu')
labels = csv_to_tensor(data_path + "test_generator/test_generator_label.csv", device='cpu')
num_mut = labels[:, -1].unsqueeze(1)

print("data loaded")

# Load model
path = "../../../trained_models/exp_generator/"
signet = SigNet(classifier=path + "classifier",
                finetuner_realistic_low=path + "finetuner_generator_low_2",
                finetuner_realistic_large=path + "finetuner_generator_large",
                errorfinder=path + "errorfinder_generator_9",
                opportunities_name_or_path=None,
                signatures_path=data_folder + "data.xlsx",
                mutation_type_order=data_folder + "mutation_type_order.xlsx")

print("model read")

finetuner_guess, upper_bound, lower_bound, classification, normalized_input = signet(inputs*labels[:, -1].reshape(-1, 1), numpy=True)

print("forwarded")

plot_weights(finetuner_guess[-1,:], upper_bound[-1,:], lower_bound[-1,:], list(pd.read_excel("../../../data/data.xlsx").columns)[1:], '')


# plot_all_metrics_vs_mutations(list_of_methods=['Baseline', 'Finetuner'],
#                               list_of_guesses=[signet.baseline_guess, finetuner_guess],
#                               label=label,
#                               show=True)

# plot_metric_vs_mutations_classifier(guess=signet.finetuner_errorfinder.classification_results,
#                                     label=torch.ones((inputs.shape[0])),
#                                     num_muts_list=label[:, -1])

# plot_metric_vs_mutations(list_of_metrics=["accuracy %", "reconstruction_error"],
#                          list_of_methods=['Baseline', 'Finetuner'],
#                          list_of_guesses=[signet.baseline_guess, finetuner_guess],
#                          label=label,
#                          show=True,
#                          signatures=signet.signatures,
#                          mutation_distributions=inputs)

# signatures = read_signatures("../../../data/data.xlsx", "../../../data/mutation_type_order.xlsx")
# reconstruction_error = get_reconstruction_error(torch.tensor(normalized_input), torch.tensor(signet.baseline_guess), signatures)
# # Write final outputs
# input_file = pd.DataFrame(normalized_input)
# input_file.set_axis(["sample_%s"%i for i in range(normalized_input.shape[0])], axis='index')
# print(input_file)
# output_path = "../../../data/exp_final/test/plots"
# write_final_outputs(finetuner_guess, lower_bound, upper_bound, signet.baseline_guess.detach().numpy(), classification, reconstruction_error.detach().numpy(), input_file, output_path)

# plot_reconstruction(inputs, finetuner_guess, signatures, list(range(0,inputs.shape[0], 1000)), output_path)
# plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, show=True)
# plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel(data_folder + "data.xlsx").columns)[1:], show=True)
# plot_interval_metrics_vs_sigs(label, upper_bound, lower_bound, '')

# errorfiner_realistic = read_model(model_path + experiment_id + "/errorfinder_realistic")
# upper_bound, lower_bound = errorfiner_realistic(finetuner_guess, label[:,-1].reshape(-1,1))

# plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, '')
# plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')