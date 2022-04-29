from email.mime import base
import os
import sys

from matplotlib.pyplot import show
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from modules.combined_finetuner import CombinedFinetuner
from utilities.io import csv_to_tensor, read_data_generator, read_signatures, read_test_data, read_model, tensor_to_csv
from utilities.plotting import plot_all_metrics_vs_mutations, plot_metric_vs_mutations, plot_metric_vs_sigs, plot_weights_comparison
from utilities.metrics import get_classification_metrics
from utilities.data_generator import DataGenerator

experiment_id = "exp_finetuner"

# Load data
test_input = csv_to_tensor("../../data/exp_oversample/test_input.csv", device="cpu")
test_label = csv_to_tensor("../../data/exp_oversample/test_label.csv", device="cpu")

# train_data, test_data = read_data_generator(device='cpu', data_id = "real_data", data_folder = "../../data/", cosmic_version = 'v3', type='real')
# test_label = test_data.inputs 
signatures = read_signatures("../../data/data.xlsx")
# data_generator = DataGenerator(signatures=signatures,
#                                    seed=None,
#                                    shuffle=True)
# test_input, test_label = data_generator.make_input(labels=test_label, set="test", large_low="large", normalize=True)
# tensor_to_csv(test_input, "../../data/exp_oversample/test_input.csv")
# tensor_to_csv(test_label, "../../data/exp_oversample/test_label.csv")

# Load Baseline and get guess
baseline = Baseline(signatures)
baseline_guess = baseline.get_weights_batch(test_input)

# Load finetuner and get predictions

models_path = "../../trained_models/%s/"%experiment_id
finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "finetuner_low",
                              large_mum_mut_dir=models_path + "finetuner_large")
finetuner_guess = finetuner(mutation_dist=test_input,
                            baseline_guess=baseline_guess,
                            num_mut=test_label[:,-1].view(-1, 1))

# tensor_to_csv(finetuner_guess, '../../data/exp_not_norm/test/test_signet_output.csv')

list_of_methods = ['baseline', 'finetuner']
list_of_guesses = [baseline_guess, finetuner_guess]

plot_all_metrics_vs_mutations(list_of_methods, list_of_guesses, test_label, '', show=True)

metrics_baseline = get_classification_metrics(label_batch=test_label[:, :-1],
                                     prediction_batch=list_of_guesses[0][:, :])
metrics_guess_1 = get_classification_metrics(label_batch=test_label[:, :-1],
                                     prediction_batch=list_of_guesses[1][:, :])

plot_weights_comparison(true_labels=metrics_baseline["MAE_sign"],
                        guessed_labels=metrics_guess_1["MAE_sign"], 
                        pred_upper=metrics_guess_1["MAE_sign"],
                        pred_lower=metrics_guess_1["MAE_sign"],
                        sigs_names=[str(v+1) for v in list(range(72))],
                        plot_path="")
                        #labels={"true":"baseline", "guessed":"finetuner"})


# list_of_metrics = ["MAE_p", "MAE_n", "fpr", "fnr"]

# plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, test_label, "../../plots/%s/metrics_vs_sigs.png"%experiment_id)
# plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, test_label, "../../plots/%s/metric_vs_mutations.png"%experiment_id)

# list_of_metrics = ["accuracy %", "sens: tp/p %", "spec: tn/n %"]

# plot_metric_vs_sigs(list_of_metrics, list_of_methods, list_of_guesses, test_label, "../../plots/%s/metrics_acc_vs_sigs.png"%experiment_id)
# plot_metric_vs_mutations(list_of_metrics, list_of_methods, list_of_guesses, test_label, show=True)


# indexes = (label_batch[:,-1]<1e3)
# label_batch = label_batch[indexes, :]
# print(label_batch)
# baseline_guess = baseline_guess[indexes, :]
# finetuner_nobaseline_guess = finetuner_nobaseline_guess[indexes, :]
# list_of_guesses = [baseline_guess, finetuner_nobaseline_guess]


# small to unknown
# list_of_guesses = [small_to_unknown(g) for g in list_of_guesses]
# print(label_batch[0])
# labels = torch.cat([small_to_unknown(label_batch[:, :-1]), label_batch[:, -1].view(-1, 1 )], dim=1)
# print(labels[0])s

    #     # Apply model to test set
    #     models_path = config["models_dir"]
    #     finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "finetuner_low",
    #                                   large_mum_mut_dir=models_path + "finetuner_large")
    #     finetuner_guess = finetuner(mutation_dist=test_input,
    #                                 baseline_guess=test_baseline,
    #                                 num_mut=test_label[:, -1].view(-1, 1))

    #     # Test model
    #     num_muts = np.unique(test_label[:, -1].detach().numpy())
    #     list_of_metrics = ["MAE", "KL", "fpr", "fnr", "accuracy %",
    #                        "precision %", "sens: tp/p %", "spec: tn/n %"]
    #     values = np.zeros((k, len(num_muts), len(list_of_metrics)))
    #     for i, num_mut in enumerate(num_muts):
    #         indexes = test_label[:, -1] == num_mut
    #         metrics = get_classification_metrics(label_batch=test_label[indexes, :-1],
    #                                              prediction_batch=finetuner_guess)
    #         for metric_index, metric in enumerate(list_of_metrics):
    #             values[k, i, metric_index] = metrics[metric]

    # # Plot final results
    # marker_size = 3
    # line_width = 0.5

    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    # axs[0, 0].plot(np.log10(num_muts), np.transpose(values[:, :, 0]),
    #                marker='o', linewidth=line_width, markersize=marker_size)
    # axs[0, 1].plot(np.log10(num_muts), np.transpose(values[:, :, 1]),
    #                marker='o', linewidth=line_width, markersize=marker_size)
    # axs[1, 0].plot(np.log10(num_muts), np.transpose(values[:, :, 2]),
    #                marker='o', linewidth=line_width, markersize=marker_size)
    # axs[1, 1].plot(np.log10(num_muts), np.transpose(values[:, :, 3]),
    #                marker='o', linewidth=line_width, markersize=marker_size)
    # plt.show()

    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    # axs[0, 0].plot(np.log10(num_muts), np.transpose(values[:, :, 4]),
    #                marker='o', linewidth=line_width, markersize=marker_size)
    # axs[0, 1].plot(np.log10(num_muts), np.transpose(values[:, :, 5]),
    #                marker='o', linewidth=line_width, markersize=marker_size)
    # axs[1, 0].plot(np.log10(num_muts), np.transpose(values[:, :, 6]),
    #                marker='o', linewidth=line_width, markersize=marker_size)
    # axs[1, 1].plot(np.log10(num_muts), np.transpose(values[:, :, 7]),
    #                marker='o', linewidth=line_width, markersize=marker_size)
    # plt.show()
    # plt.close()
