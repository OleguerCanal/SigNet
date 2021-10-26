import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.combined_finetuner import CombinedFinetuner
from models.baseline import Baseline
from utilities.io import read_signatures, read_test_data
from utilities.plotting import plot_metric_vs_mutations, plot_metric_vs_sigs

experiment_id = "exp_v2"

test_id = "test"

large_mum_mut_dir = "../../trained_models/%s/finetuner_realistic_large_v2"%experiment_id
low_mum_mut_dir = "../../trained_models/%s/finetuner_realistic_low_v2"%experiment_id

input_batch, label_batch = read_test_data("cpu", experiment_id, test_id, data_folder="../../data")
signatures = read_signatures("../../data/data_v2.xlsx")

baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch, n_workers=1)

finetuner = CombinedFinetuner(low_mum_mut_dir=low_mum_mut_dir, large_mum_mut_dir=large_mum_mut_dir)

finetuner_guess, ind_order = finetuner(input_batch, baseline_guess, label_batch[:,-1])
label = label_batch[ind_order,:]
baseline_guess = baseline_guess[ind_order,:]

list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baseline_guess, finetuner_guess]

list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_sigs.png"%experiment_id)
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_muts.png"%experiment_id)

list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_sigs.png"%experiment_id)
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_muts.png"%experiment_id)
