import os

import numpy as np
import pandas as pd
import torch
import seaborn as sn

from model import SignatureNet
from utilities.dataloader import DataLoader
from utilities.metrics import get_cosine_similarity, get_entropy, get_cross_entropy, get_wasserstein_distance
from utilities.plotting import plot_signature, plot_confusion_matrix


class ModelTester:
    def __init__(self, dataloader, num_classes):
        self.num_classes = num_classes
        self.dataloader = dataloader

    def test(self, model):
        input_batch, label_batch = self.dataloader.get_batch()

        guessed_labels = model(input_batch)

        cross_entropy = get_cross_entropy(guessed_labels, label_batch)
        kl_divergence = cross_entropy - get_entropy(label_batch)
        cosine_similarity = get_cosine_similarity(guessed_labels, label_batch)
        wasserstein_distance = get_wasserstein_distance(
            guessed_labels, label_batch)

        print("cross_entropy:", np.round(cross_entropy.item(), decimals=3))
        print("kl_divergence:", np.round(kl_divergence.item(), decimals=3))
        print("cosine_similarity:", np.round(cosine_similarity.item(), decimals=3))
        print("wasserstein_distance:", np.round(wasserstein_distance, decimals=3))

        label_sigs, predicted_sigs = self.probs_batch_to_sigs(
            label_batch, guessed_labels, cutoff=0.05, num_classes=self.num_classes)
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
    experiment_id = "test_0"

    # Model params
    num_hidden_layers = 4
    num_neurons = 500
    num_classes = 10

    model = SignatureNet(num_classes=num_classes,
                         num_hidden_layers=num_hidden_layers,
                         num_units=num_neurons)
    model.load_state_dict(torch.load(os.path.join("models", experiment_id)))
    model.eval()

    data = pd.read_excel("data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:model.num_classes]
    dataloader = DataLoader(signatures=signatures,
                             batch_size=500,
                             n_samples=5000,
                             min_n_signatures=1,
                             max_n_signatures=5)

    model_tester = ModelTester(dataloader=dataloader,
                               num_classes=model.num_classes)
    model_tester.test(model=model)
