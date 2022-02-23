import os
import sys

from matplotlib.pyplot import show
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from modules.combined_finetuner import CombinedFinetuner
from utilities.io import read_signatures, read_test_data, read_model, tensor_to_csv
from utilities.plotting import plot_all_metrics_vs_mutations, plot_metric_vs_mutations, plot_metric_vs_sigs, plot_weights_comparison
from utilities.metrics import get_classification_metrics


experiment_id = "exp_not_norm"
test_id = "test"
# finetuner_directory = "../../trained_models/%s/finetuner_generator_low"%experiment_id
# finetuner_directory = "../../trained_models/%s/finetuner_nobaseline"%experiment_id

# Load data
# inputs = csv_to_tensor(path + "/%s_input.csv" % (test_id), device=device)

input_batch, label_batch = read_test_data("cpu", experiment_id, test_id, data_folder="../../data")
signatures = read_signatures("../../data/data.xlsx")

# Load Baseline and get guess
baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)

# Load finetuner and get predictions
# finetuner = read_model(finetuner_directory)
# finetuner_guess_01 = finetuner(mutation_dist=input_batch,
#                             baseline_guess=baseline_guess,
#                             num_mut=label_batch[:,-1].view(-1, 1))

models_path = "../../trained_models/%s/"%experiment_id
finetuner_nobaseline = CombinedFinetuner(low_mum_mut_dir=models_path + "finetuner_not_norm_no_baseline_low",
                                         large_mum_mut_dir=models_path + "finetuner_not_norm_no_baseline_large")
finetuner_nobaseline_guess = finetuner_nobaseline(mutation_dist=input_batch,
                                                  num_mut=label_batch[:,-1].view(-1, 1))

tensor_to_csv(finetuner_nobaseline_guess, '../../data/exp_not_norm/test/test_signet_output.csv')
# list_of_methods = ['baseline', 'finetuner_nobaseline']
# list_of_guesses = [baseline_guess, finetuner_nobaseline_guess]

list_of_methods = ['baseline', 'finetuner_nobaseline']
list_of_guesses = [baseline_guess, finetuner_nobaseline_guess]

# small to unknown
# list_of_guesses = [small_to_unknown(g) for g in list_of_guesses]
# print(label_batch[0])
# labels = torch.cat([small_to_unknown(label_batch[:, :-1]), label_batch[:, -1].view(-1, 1 )], dim=1)
# print(labels[0])

plot_all_metrics_vs_mutations(list_of_methods, list_of_guesses, label_batch, '', show=True)

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


# # list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

# # plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metrics_vs_sigs.png"%experiment_id)
# # plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metric_vs_mutations.png"%experiment_id)

# # list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

# # plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metrics_acc_vs_sigs.png"%experiment_id)
# # plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label_batch, "../../plots/%s/metric_acc_vs_mutations.png"%experiment_id)
