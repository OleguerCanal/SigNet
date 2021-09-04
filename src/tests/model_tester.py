import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sn


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.yapsa_inspired_baseline import YapsaInspiredBaseline
from utilities.io import read_test_data
from models.error_finder import ErrorFinder
from models.finetuner import FineTuner
from utilities.dataloader import DataLoader
from utilities.metrics import *
from utilities.plotting import plot_interval_width_vs_mutations, plot_signature, plot_confusion_matrix, plot_weights, plot_weights_comparison, plot_interval_performance


class ModelTester:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def test(self, guessed_labels, true_labels):
        mse = get_MSE(guessed_labels, true_labels)
        cross_entropy = get_cross_entropy(guessed_labels, true_labels)
        kl_divergence = get_kl_divergence(guessed_labels, true_labels)
        js_divergence = get_jensen_shannon(guessed_labels, true_labels)
        cosine_similarity = get_cosine_similarity(guessed_labels, true_labels)
        wasserstein_distance = get_wasserstein_distance(
            guessed_labels, true_labels)

        print("mse:", np.round(mse.item(), decimals=5))
        print("cross_entropy:", np.round(cross_entropy.item(), decimals=3))
        print("kl_divergence:", np.round(kl_divergence.item(), decimals=3))
        print("js_divergence:", np.round(js_divergence.item(), decimals=3))
        print("cosine_similarity:", np.round(
            cosine_similarity.item(), decimals=3))
        print("wasserstein_distance:", np.round(
            wasserstein_distance.item(), decimals=3))

        label_sigs, predicted_sigs = self.probs_batch_to_sigs(
            true_labels, guessed_labels, cutoff=0.05, num_classes=self.num_classes)

        conf_mat = plot_confusion_matrix(
            label_sigs, predicted_sigs, range(self.num_classes+1))

    def probs_batch_to_sigs(self, label_batch, predicted_batch, cutoff, num_classes):
        label_sigs_list = torch.zeros(0, dtype=torch.long)
        predicted_sigs_list = torch.zeros(0, dtype=torch.long)
        for i in range(len(label_batch)):
            for j in range(len(label_batch[i])):
                if label_batch[i][j] > cutoff and predicted_batch[i][j] > cutoff:
                    label_sigs_list = torch.cat(
                        [label_sigs_list, torch.from_numpy(np.array([j]))])
                    predicted_sigs_list = torch.cat(
                        [predicted_sigs_list, torch.from_numpy(np.array([j]))])
                if label_batch[i][j] > cutoff and predicted_batch[i][j] < cutoff:
                    label_sigs_list = torch.cat(
                        [label_sigs_list, torch.from_numpy(np.array([j]))])
                    predicted_sigs_list = torch.cat(
                        [predicted_sigs_list, torch.from_numpy(np.array([num_classes]))])
                if label_batch[i][j] < cutoff and predicted_batch[i][j] > cutoff:
                    label_sigs_list = torch.cat(
                        [label_sigs_list, torch.from_numpy(np.array([num_classes]))])
                    predicted_sigs_list = torch.cat(
                        [predicted_sigs_list, torch.from_numpy(np.array([j]))])
        return label_sigs_list, predicted_sigs_list


if __name__ == "__main__":
    experiment_id = "exp_0"
    test_id = "test_realistic"
    device = "cpu"

    # Model params finetuner
    model_id_finetuner = "finetuner_realistic"
    num_hidden_layers = 2
    num_neurons = 1300
    num_classes = 72

    # Model params error
    model_id_error_learner = "errorfinder_realistic"
    num_hidden_layers_pos = 1
    num_neurons_pos = 1000
    num_hidden_layers_neg = 1
    num_neurons_neg = 1000

    # Open data
    data = pd.read_excel("../../data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]

    input_batch, label_mut_batch = read_test_data(device=device,
                                                  experiment_id=experiment_id,
                                                  test_id=test_id,
                                                  data_folder="../../data")
    label_batch = label_mut_batch[:, :-1]
    num_mut = label_mut_batch[:, -1].reshape((-1, 1))

    # Baseline:
    sf = YapsaInspiredBaseline(signatures)
    baseline_batch = sf.get_weights_batch(input_batch)  # [:50, ...])

    # Instantiate model and do predictions for finetuner:
    model = FineTuner(num_classes=num_classes,
                      num_hidden_layers=num_hidden_layers,
                      num_units=num_neurons)
    model.load_state_dict(torch.load(os.path.join(
        "../../trained_models/" + experiment_id, model_id_finetuner), map_location=torch.device(device)))
    model.eval()
    guessed_labels = model(input_batch, baseline_batch, num_mut)

    # Instantiate model and do predictions for error learner:
    model_error = ErrorFinder(num_classes=num_classes,
                              num_hidden_layers_pos=num_hidden_layers_pos,
                              num_units_pos=num_neurons_pos,
                              num_hidden_layers_neg=num_hidden_layers_neg,
                              num_units_neg=num_neurons_neg)

    model_error.load_state_dict(torch.load(os.path.join(
        "../../trained_models/" + experiment_id, model_id_error_learner), map_location=torch.device(device)))
    model_error.eval()
    pred_upper, pred_lower = model_error(guessed_labels, num_mut)

    # # Get metrics
    # model_tester = ModelTester(num_classes=num_classes)
    # model_tester.test(guessed_labels=guessed_labels, true_labels=label_batch)

    # # Plot signatures
    # plot_weights_comparison(label_batch[0, :].detach().numpy(), guessed_labels[0, :].detach().numpy(
    # ), pred_upper[0, :].detach().numpy(),pred_lower[0, :].detach().numpy(), list(data.columns)[2:])
    # plot_weights_comparison(label_batch[5000, :].detach().numpy(), guessed_labels[5000, :].detach().numpy(
    # ), pred_upper[5000, :].detach().numpy(),pred_lower[5000, :].detach().numpy(), list(data.columns)[2:])
    # plot_weights_comparison(label_batch[9000, :].detach().numpy(), guessed_labels[9000, :].detach().numpy(
    # ), pred_upper[9000, :].detach().numpy(),pred_lower[9000, :].detach().numpy(), list(data.columns)[2:])

    # Plot interval performance
    plot_interval_performance(label_batch, pred_upper,pred_lower, list(data.columns)[2:])

    # Plot interval width vs number of mutations
    plot_interval_width_vs_mutations(
        pred_upper, pred_lower, list(data.columns)[2:])

    # Print metrics
    # in_prop, mean_length = get_pi_metrics(
    #     label_batch, pred_lower=pred_lower, pred_upper=pred_upper)
    # print("In proportion:", in_prop.detach().numpy())
    # print("Mean length:", mean_length.detach().numpy())
