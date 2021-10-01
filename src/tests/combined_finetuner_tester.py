import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.combined_finetuner import CombinedFinetuner
from models.baseline import Baseline
from utilities.io import read_signatures, read_test_data
from utilities.plotting import plot_metric_vs_mutations, plot_metric_vs_sigs

experiment_id = "exp_random_2_nets"

finetuner_model_name_low = "finetuner_random_low_random_low"
finetuner_params_low = {"num_hidden_layers": 2,
                        "num_units": 600,
                        "num_classes": 72,
                        "sigmoid_params": [500,150]}

finetuner_model_name_large = "finetuner_1.0_large_mixture_1.0_large"
finetuner_params_large = {"num_hidden_layers": 2,
                          "num_units": 600,
                          "num_classes": 72,
                          "sigmoid_params": [50000, 10000]}

test_id = "test_random"

input_batch, label_batch = read_test_data("cpu", experiment_id, test_id, data_folder="../../data")
signatures = read_signatures("../../data/data.xlsx")

baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)

finetuner = CombinedFinetuner(experiment_id,
                finetuner_model_name_low,
                finetuner_params_low,
                finetuner_model_name_large,
                finetuner_params_large)

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
