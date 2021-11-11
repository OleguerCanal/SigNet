import os
import sys

import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.signet import SigNet
from utilities.io import read_methods_guesses, read_signatures, read_test_data, read_model
from utilities.plotting import plot_all_metrics_vs_mutations, plot_interval_metrics_vs_mutations, plot_interval_metrics_vs_sigs, plot_interval_performance, plot_metric_vs_mutations, plot_metric_vs_sigs

# Read data
test_id = "test"
input_batch, label = read_test_data('cpu', "exp_final", test_id, data_folder="../../data")

# Load model
path = "../../trained_models/exp_final/"
signet = SigNet(classifier=path + "classifier",
                finetuner_random_low=path + "finetuner_random_low",
                finetuner_random_large=path + "finetuner_random_large",
                finetuner_realistic_low=path + "finetuner_realistic_low",
                finetuner_realistic_large=path + "finetuner_realistic_large",
                errorfinder_random_low=path + "errorfinder_random_low",
                errorfinder_random_large=path + "errorfinder_random_large",
                errorfinder_realistic_low=path + "errorfinder_realistic_low",
                errorfinder_realistic_large=path + "errorfinder_realistic_large",
                path_opportunities=None,
                signatures_path="../../data/data.xlsx")

finetuner_guess, upper_bound, lower_bound = signet(input_batch, numpy=False)

# list_of_methods = ["decompTumor2Sig", "mutSignatures", "SignatureEstimationQP","YAPSA", "deconstructSigs"]#, "MutationalPatterns",
# list_of_guesses, label = read_methods_guesses('cpu', experiment_id, "test", list_of_methods, data_folder="../../data")

# list_of_methods += ['Baseline', 'Finetuner']
# list_of_guesses += [baseline_guess, finetuner_guess]

# plot_all_metrics_vs_mutations(list_of_methods, list_of_guesses, label, '')
plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, '')
plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')
# plot_interval_metrics_vs_sigs(label, upper_bound, lower_bound, '')

# errorfiner_realistic = read_model(model_path + experiment_id + "/errorfinder_realistic")
# upper_bound, lower_bound = errorfiner_realistic(finetuner_guess, label[:,-1].reshape(-1,1))

# plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, '')
# plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')