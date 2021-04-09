import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sn

from baseline import SignatureFinder
from model import SignatureNet
from utilities.dataloader import DataLoader
from utilities.metrics import *
from utilities.plotting import plot_signature, plot_confusion_matrix, plot_weights, plot_weights_comparison, plot_weights_comparison_deconstructSigs


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
        print("cosine_similarity:", np.round(cosine_similarity.item(), decimals=3))
        print("wasserstein_distance:", np.round(wasserstein_distance.item(), decimals=3))
        
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
    # Model params
    experiment_id = "learn_error_5"
    num_hidden_layers = 4
    num_neurons = 600
    num_classes = 72

    # Generate data
    data = pd.read_excel("data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                 for i in range(2, 74)][:num_classes]

    # input_batch = torch.tensor(pd.read_csv("data/test_input_w01.csv", header=None).values, dtype=torch.float)
    # label_mut_batch = torch.tensor(pd.read_csv("data/test_label_w01.csv", header=None).values, dtype=torch.float)
    # label_batch = label_mut_batch[:,:num_classes]
    # num_mut = torch.reshape(label_mut_batch[:,num_classes], (list(label_mut_batch.size())[0],1))

    # baseline_batch = torch.tensor(pd.read_csv("data/test_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    # deconstructSigs_batch = torch.tensor(pd.read_csv("data/deconstructSigs_test_w01.csv", header=None).values, dtype=torch.float)
    
    num_mut = torch.tensor(pd.read_csv("data/MC3_ACC_num_mut.csv", header=None).values, dtype=torch.float)
    num_mut = torch.reshape(num_mut, (list(num_mut.size())[0],1))

    baseline_batch = torch.tensor(pd.read_csv("data/MC3_ACC_data_baseline_JS.csv", header=None).values, dtype=torch.float)
    deconstructSigs_batch = torch.tensor(pd.read_csv("data/MC3_ACC_deconstructSigs.csv", header=None).values, dtype=torch.float)
    
    # Instantiate model and do predictions
    model = SignatureNet(signatures=signatures,
                          num_classes=num_classes,
                          num_hidden_layers=num_hidden_layers,
                          num_units=num_neurons)
    model.load_state_dict(torch.load(os.path.join("models", experiment_id)))
    model.eval()
    guessed_error = model(baseline_batch, num_mut)
    

    # Plot signatures

    plot_weights_comparison(deconstructSigs_batch[0,:].detach().numpy(), baseline_batch[0,:].detach().numpy(), guessed_error[0,:].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison(deconstructSigs_batch[9,:].detach().numpy(),baseline_batch[9,:].detach().numpy(), guessed_error[9,:].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison(deconstructSigs_batch[22,:].detach().numpy(),baseline_batch[22,:].detach().numpy(), guessed_error[22,:].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison(deconstructSigs_batch[-3,:].detach().numpy(),baseline_batch[-3,:].detach().numpy(), guessed_error[-3,:].detach().numpy(), list(data.columns)[2:])
