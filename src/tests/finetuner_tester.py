import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from utilities.io import read_signatures, read_test_data, read_model
from utilities.plotting import plot_all_metrics_vs_mutations, plot_metric_vs_mutations, plot_metric_vs_sigs

experiment_id = "exp_final"
test_id = "test_random"
finetuner_directory = "../../trained_models/%s/01fn_0001fp_kl_random"%experiment_id

# Load data
input_batch, label_batch = read_test_data("cpu", experiment_id, test_id, data_folder="../../data")
signatures = read_signatures("../../data/data.xlsx")

# Load Baseline and get guess
baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)

# Load finetuner and get predictions
finetuner = read_model(finetuner_directory)
finetuner_guess_01 = finetuner(mutation_dist=input_batch,
                            baseline_guess=baseline_guess,
                            num_mut=label_batch[:,-1].view(-1, 1))


# Plot results
# list_of_methods = ["decompTumor2Sig", "MutationalPatterns", "mutSignatures", "SignatureEstimationQP","YAPSA"]#, "deconstructSigs"]
# list_of_guesses, label = read_methods_guesses('cpu', experiment_id, "test", list_of_methods, data_folder="../../data")

list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baseline_guess, finetuner_guess_01]

plot_all_metrics_vs_mutations( list_of_methods, list_of_guesses, label_batch, '')

# list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

# plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metrics_vs_sigs.png"%experiment_id)
# plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metric_vs_mutations.png"%experiment_id)

# list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

# plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metrics_acc_vs_sigs.png"%experiment_id)
# plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metric_acc_vs_mutations.png"%experiment_id)
