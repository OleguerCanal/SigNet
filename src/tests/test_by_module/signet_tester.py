import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.signet import SigNet
from utilities.io import read_signatures, read_test_data, csv_to_tensor, write_final_outputs
from utilities.plotting import plot_all_metrics_vs_mutations, plot_interval_metrics_vs_mutations, plot_interval_metrics_vs_sigs, plot_interval_performance, plot_metric_vs_mutations, plot_metric_vs_sigs, plot_metric_vs_mutations_classifier, plot_reconstruction
from utilities.metrics import get_reconstruction_error

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
path = "../../../trained_models/exp_final_3/"
signet = SigNet(classifier=path + "classifier",
                finetuner_random_low=path + "finetuner_random_low",
                finetuner_random_large=path + "finetuner_random_large",
                finetuner_realistic_low=path + "finetuner_realistic_low",
                finetuner_realistic_large=path + "finetuner_realistic_large",
                errorfinder=path + "errorfinder",
                opportunities_name_or_path=None,
                signatures_path=data_folder + "data.xlsx",
                mutation_type_order=data_folder + "mutation_type_order.xlsx")

print("model read")

finetuner_guess, upper_bound, lower_bound, classification, normalized_input = signet(inputs*label[:, -1].reshape(-1, 1), numpy=True)

print("forwarded")
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

signatures = read_signatures("../../../data/data.xlsx", "../../../data/mutation_type_order.xlsx")
reconstruction_error = get_reconstruction_error(torch.tensor(normalized_input), torch.tensor(signet.baseline_guess), signatures)
# Write final outputs
input_file = pd.DataFrame(normalized_input)
input_file.set_axis(["sample_%s"%i for i in range(normalized_input.shape[0])], axis='index')
print(input_file)
output_path = "../../../data/exp_final/test/plots"
write_final_outputs(finetuner_guess, lower_bound, upper_bound, signet.baseline_guess.detach().numpy(), classification, reconstruction_error.detach().numpy(), input_file, output_path)

plot_reconstruction(inputs, finetuner_guess, signatures, list(range(0,inputs.shape[0], 1000)), output_path)
# plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, show=True)
# plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel(data_folder + "data.xlsx").columns)[1:], show=True)
# plot_interval_metrics_vs_sigs(label, upper_bound, lower_bound, '')

# errorfiner_realistic = read_model(model_path + experiment_id + "/errorfinder_realistic")
# upper_bound, lower_bound = errorfiner_realistic(finetuner_guess, label[:,-1].reshape(-1,1))

# plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, '')
# plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')