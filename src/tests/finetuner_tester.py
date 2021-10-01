import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.combined_finetuner import CombinedFinetuner
from models.yapsa_inspired_baseline import YapsaInspiredBaseline
from utilities.io import read_signatures, read_test_data, read_model
from utilities.plotting import plot_metric_vs_mutations, plot_metric_vs_sigs

experiment_id = "exp_0"
test_id = "test_random"
finetuner_directory = "../../trained_models/exp_0/finetuner_random_test_new_saving"

# Load data
input_batch, label_batch = read_test_data("cpu", experiment_id, test_id, data_folder="../../data")
signatures = read_signatures("../../data/data.xlsx")

# Load Baseline and get guess
baseline = YapsaInspiredBaseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)

# Load finetuner and get predictions
finetuner = read_model(finetuner_directory)
finetuner_guess = finetuner(mutation_dist=input_batch,
                            weights=baseline_guess,
                            num_mut=label_batch[:,-1].view(-1, 1))

# Plot results
list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baseline_guess, finetuner_guess]

list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/random_vs_sigs.png"%experiment_id)
