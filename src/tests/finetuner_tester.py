import os
import sys

from matplotlib.pyplot import show
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from modules.combined_finetuner import CombinedFinetuner
from utilities.io import read_signatures, read_test_data, read_model
from utilities.plotting import plot_all_metrics_vs_mutations, plot_metric_vs_mutations, plot_metric_vs_sigs, plot_weights_comparison
from utilities.metrics import get_classification_metrics

experiment_id = "exp_generator"
test_id = "test_generator"
finetuner_directory = "../../trained_models/%s/finetuner_generator_low_2"%experiment_id
# finetuner_directory2 = "../../trained_models/%s/finetuner_generator_residual_nosoft_large"%experiment_id

# Load data
input_batch, label_batch = read_test_data("cpu", experiment_id, test_id, data_folder="../../data")
signatures = read_signatures("../../data/data.xlsx")

# Load Baseline and get guess
baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)

# Load finetuner and get predictions
# finetuner = read_model(finetuner_directory)
models_path =  "../../trained_models/%s/"%experiment_id
finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "finetuner_generator_low_2",
                              large_mum_mut_dir=models_path + "finetuner_generator_large")
finetuner_guess_01 = finetuner(mutation_dist=input_batch,
                                baseline_guess=baseline_guess,
                                num_mut=label_batch[:,-1].view(-1, 1))

# Plot results
# list_of_methods = ["decompTumor2Sig", "MutationalPatterns", "mutSignatures", "SignatureEstimationQP","YAPSA"]#, "deconstructSigs"]
# list_of_guesses, label = read_methods_guesses('cpu', experiment_id, "test", list_of_methods, data_folder="../../data")

list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baseline_guess, finetuner_guess_01]

# plot_all_metrics_vs_mutations(list_of_methods, list_of_guesses, label_batch, '', show=True)

# indexes = label_batch[:, -1] >= 1e4
metrics_baseline = get_classification_metrics(label_batch=label_batch[:, :-1],
                                     prediction_batch=list_of_guesses[0][:, :])
metrics_guess_1 = get_classification_metrics(label_batch=label_batch[:, :-1],
                                     prediction_batch=list_of_guesses[1][:, :])

plot_weights_comparison(true_labels=metrics_baseline["MAE_sign"],
                        guessed_labels=metrics_guess_1["MAE_sign"], 
                        pred_upper=metrics_guess_1["MAE_sign"],
                        pred_lower=metrics_guess_1["MAE_sign"],
                        sigs_names=[str(v+1) for v in list(range(72))],
                        plot_path="")


# list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

# plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metrics_vs_sigs.png"%experiment_id)
# plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metric_vs_mutations.png"%experiment_id)

# list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

# plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metrics_acc_vs_sigs.png"%experiment_id)
# plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metric_acc_vs_mutations.png"%experiment_id)
