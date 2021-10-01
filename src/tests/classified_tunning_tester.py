import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.classified_tunning import ClassifiedFinetuner
from models.baseline import Baseline
from utilities.io import csv_to_tensor, read_signatures, read_test_data
from utilities.plotting import plot_metric_vs_mutations, plot_metric_vs_sigs

experiment_id = "exp_random_2_nets"

model_path = "../../trained_models/"

classifier_exp = "exp_classifier/"
classifier_model = model_path + classifier_exp + "classifier"

random_exp = "exp_0/"
random_finetuner_model = model_path + random_exp + "finetuner_random"

realistic_exp = "exp_0/"
realistic_finetuner_model = model_path + realistic_exp + "finetuner_realistic"

experiment_id = "exp_classifier"

test_id = "test_mixed"
input_batch, label = read_test_data('cpu', 'exp_0', test_id, data_folder="../../data")

signatures = read_signatures("../../data/data.xlsx")

baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)

finetuner = ClassifiedFinetuner(classifier_model,
                 realistic_finetuner_model,
                 random_finetuner_model)

finetuner_guess = finetuner(input_batch, baseline_guess, label[:,-1].reshape(-1,1))

list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baseline_guess, finetuner_guess]

list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_sigs.png"%experiment_id)
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_muts.png"%experiment_id)

list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_sigs.png"%experiment_id)
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_muts.png"%experiment_id)
