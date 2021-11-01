import os
import sys

import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.classified_tunning import ClassifiedFinetuner
from modules.combined_finetuner import CombinedFinetuner
from models.baseline import Baseline
from models.error_finder import ErrorFinder
from modules.classified_tunning_error import ClassifiedFinetunerErrorfinder
from utilities.io import csv_to_tensor, read_methods_guesses, read_signatures, read_test_data, read_model
from utilities.plotting import plot_all_metrics_vs_mutations, plot_interval_metrics_vs_mutations, plot_interval_metrics_vs_sigs, plot_interval_performance, plot_metric_vs_mutations, plot_metric_vs_sigs
from modules.combined_errorfinder import CombinedErrorfinder


model_path = "../../trained_models/"
experiment_id = "exp_final"

# Model ids
classifier = model_path + "classifier"
realistic_finetuner_low_nummut = model_path + experiment_id + "/finetuner_realistic_low"
realistic_finetuner_large_nummut = model_path + experiment_id + "/finetuner_realistic_large"
random_finetuner_low_nummut = model_path + experiment_id + "/finetuner_random_low"
random_finetuner_large_nummut = model_path + experiment_id + "/finetuner_random_large"
random_errorfinder_low = model_path + experiment_id + "/errorfinder_random_low"
random_errorfinder_large = model_path + experiment_id + "/errorfinder_random_large"
realistic_errorfinder_low = model_path + experiment_id + "/errorfinder_realistic_low"
realistic_errorfinder_large = model_path + experiment_id + "/errorfinder_realistic_large"

test_id = "test"
input_batch, label = read_test_data('cpu', experiment_id, test_id, data_folder="../../data")

signatures = read_signatures("../../data/data.xlsx")

baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch, n_workers=1)

realistic_finetuner = CombinedFinetuner(low_mum_mut_dir=realistic_finetuner_low_nummut,
                                        large_mum_mut_dir=realistic_finetuner_large_nummut)

random_finetuner = CombinedFinetuner(low_mum_mut_dir=random_finetuner_low_nummut,
                                        large_mum_mut_dir=random_finetuner_large_nummut)

realistic_errorfinder = CombinedErrorfinder(low_mum_mut_dir=realistic_errorfinder_low,
                                        large_mum_mut_dir=realistic_errorfinder_large)

random_errorfinder = CombinedErrorfinder(low_mum_mut_dir=random_errorfinder_low,
                                        large_mum_mut_dir=random_errorfinder_large)

finetuner_errorfinder = ClassifiedFinetunerErrorfinder(classifier=read_model(classifier),
                                realistic_finetuner=realistic_finetuner,
                                random_finetuner=random_finetuner,
                                realistic_errorfinder=realistic_errorfinder,
                                random_errorfinder=random_errorfinder)

finetuner_guess, upper_bound, lower_bound = finetuner_errorfinder(input_batch, baseline_guess, label[:,-1].reshape(-1,1))

list_of_methods = ["decompTumor2Sig", "mutSignatures", "SignatureEstimationQP","YAPSA", "deconstructSigs"]#, "MutationalPatterns",
list_of_guesses, label = read_methods_guesses('cpu', experiment_id, "test", list_of_methods, data_folder="../../data")

list_of_methods += ['Baseline', 'Finetuner']
list_of_guesses += [baseline_guess, finetuner_guess]

# plot_all_metrics_vs_mutations(list_of_methods, list_of_guesses, label, '')
plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, '')
plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')
# plot_interval_metrics_vs_sigs(label, upper_bound, lower_bound, '')

# errorfiner_realistic = read_model(model_path + experiment_id + "/errorfinder_realistic")
# upper_bound, lower_bound = errorfiner_realistic(finetuner_guess, label[:,-1].reshape(-1,1))

# plot_interval_metrics_vs_mutations(label, upper_bound, lower_bound, '')
# plot_interval_performance(label, upper_bound, lower_bound, list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')