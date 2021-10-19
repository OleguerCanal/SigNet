import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from utilities.io import read_signatures, read_cosmic_v2_signatures, read_test_data
from utilities.plotting import plot_metric_vs_mutations, plot_metric_vs_sigs

data_id = "exp_v2"
test_id = "test_random"

large_mum_mut_dir = "../../trained_models/exp_mixture/finetuner_realistic_large_nummut"
low_mum_mut_dir = "../../trained_models/exp_0/finetuner_realistic"

input_batch, label_batch = read_test_data("cpu", data_id, test_id, data_folder="../../data")
signatures = read_cosmic_v2_signatures("../../data/data_v2.xlsx")

baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(input_batch)
label = label_batch

print(baseline_guess.shape)
print(label.shape)

list_of_methods = ['baseline']
list_of_guesses = [baseline_guess]

# list_of_metrics = ["MAE_p", "MAE_n", "fp", "fn"]

# # plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_sigs.png"%data_id)
# plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_vs_muts.png"%data_id)

# list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

# # plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_sigs.png"%data_id)
# plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, label, "../../plots/%s/random_accuracy_vs_muts.png"%data_id)




import matplotlib.pyplot as plt
import numpy as np

x = np.log10(label[:,-1].detach().numpy())
plt.hist(x, bins = 11)
plt.show()
