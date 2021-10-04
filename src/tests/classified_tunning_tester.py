import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.classified_tunning import ClassifiedFinetuner
from modules.combined_finetuner import CombinedFinetuner
from models.baseline import Baseline
from utilities.io import csv_to_tensor, read_signatures, read_test_data, read_model
from utilities.plotting import plot_metric_vs_mutations, plot_metric_vs_sigs


model_path = "../../trained_models/"
experiment_id = "exp_classifier"

# Model ids
classifier = model_path + "exp_classifier/classifier"
realistic_finetuner_low_nummut = model_path + "exp_0/finetuner_realistic"
realistic_finetuner_large_nummut = model_path + "exp_mixture/finetuner_realistic_large_nummut"
random_finetuner_low_nummut = model_path + "exp_0/finetuner_random"
random_finetuner_large_nummut = model_path + "exp_0/finetuner_random"


test_id = "test_realistic"
input_batch, label = read_test_data('cpu', 'exp_0', test_id, data_folder="../../data")

signatures = read_signatures("../../data/data.xlsx")

baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)

realistic_finetuner = CombinedFinetuner(low_mum_mut_dir=realistic_finetuner_low_nummut,
                                        large_mum_mut_dir=realistic_finetuner_large_nummut)

random_finetuner = CombinedFinetuner(low_mum_mut_dir=random_finetuner_low_nummut,
                                        large_mum_mut_dir=random_finetuner_large_nummut)

finetuner = ClassifiedFinetuner(classifier=read_model(classifier),
                                realistic_finetuner=realistic_finetuner,
                                random_finetuner=random_finetuner)

finetuner_guess = finetuner(input_batch, baseline_guess, label[:,-1].reshape(-1,1))

list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baseline_guess, finetuner_guess]

list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_sigs.png"%experiment_id)
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_muts.png"%experiment_id)

list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_sigs.png"%experiment_id)
plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_muts.png"%experiment_id)
