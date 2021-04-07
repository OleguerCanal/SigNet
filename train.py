import os
import pathlib

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SignatureNet
from model_tester import ModelTester
from utilities.dataloader import DataLoader
from utilities.metrics import *
from utilities.plotting import plot_signature, plot_confusion_matrix

# Model params
num_hidden_layers = 4
num_neurons = 600
num_classes = 72

# Training params
experiment_id = "comb_js_error_1"
iterations = 1500
batch_size = 50
num_samples = 1000
intial_learning_rate = 0.01
learning_rate_steps = 300
learning_rate_gamma = 0.7
l1_lambda = 0
FN_param = 0.1

if __name__ == "__main__":

    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    data = pd.read_excel("data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(
        torch.float32) for i in range(2, 74)][:num_classes]

    validation_input = torch.tensor(pd.read_csv("data/validation_input.csv", header=None).values, dtype=torch.float)
    validation_label = torch.tensor(pd.read_csv("data/validation_label.csv", header=None).values, dtype=torch.float)
    validation_baseline = torch.tensor(pd.read_csv("data/validation_baseline_JS.csv", header=None).values, dtype=torch.float)
    training_baseline = torch.tensor(pd.read_csv("data/training_baseline_JS.csv", header=None).values, dtype=torch.float)
    training_input = torch.tensor(pd.read_csv("data/training_input.csv", header=None).values, dtype=torch.float)
    training_label = torch.tensor(pd.read_csv("data/training_label.csv", header=None).values, dtype=torch.float)
    
    data_loader = DataLoader(signatures=signatures,
                             batch_size=batch_size,
                             n_samples=num_samples,
                             min_n_signatures=1,
                             max_n_signatures=10)

    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_id))
    writer_validation = SummaryWriter(log_dir=os.path.join("runs/validation", experiment_id))

    sn = SignatureNet(signatures=signatures,
                      num_classes=num_classes,
                      num_hidden_layers=num_hidden_layers, num_units=num_neurons)
    optimizer = optim.Adam(sn.parameters(), lr=intial_learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=learning_rate_steps, gamma=learning_rate_gamma)
    #loss = nn.CrossEntropyLoss()

    predicted_list = torch.zeros(0, dtype=torch.long)
    label_list = torch.zeros(0, dtype=torch.long)

    last_ind = 0
    for iteration in tqdm(range(int(iterations))):
        input_batch, label_batch, baseline_batch, last_ind = data_loader.select_batch( training_input, training_label, training_baseline, last_ind)
        optimizer.zero_grad()

        predicted_batch, predicted_error = sn(input_batch, baseline_batch)

        mt = ModelTester(num_classes)
        label_sigs_list, predicted_sigs_list = mt.probs_batch_to_sigs(label_batch, predicted_batch, 0.05, num_classes)
        # l = get_cross_entropy(predicted_batch, label_batch)
        # l = get_MSE(predicted_batch, label_batch)
        # l = get_kl_divergence(predicted_batch, label_batch)
        # l = get_jensen_shannon(predicted_batch, label_batch)
        l_1 = get_kl_divergence(predicted_batch, label_batch)
        l_2 = get_MSE(predicted_error, abs(predicted_batch-label_batch))
        l = l_1 + l_2
        FN = sum(predicted_sigs_list == num_classes)
        print(FN)
        l1_norm = sum(p.abs().sum() for p in sn.parameters())
        #l = l + l1_lambda*l1_norm + FN_param*FN

        l.backward()
        optimizer.step()
        scheduler.step()

        validation_prediction, val_error = sn(validation_input, validation_baseline)
        writer.add_scalar(f'metrics/cross-entropy', get_cross_entropy2(predicted_batch, label_batch), iteration)
        writer_validation.add_scalar(f'metrics/cross-entropy', get_cross_entropy2(validation_prediction, validation_label), iteration)
        writer.add_scalar(f'metrics/cosine_similarity', get_cosine_similarity(predicted_batch, label_batch), iteration)
        writer_validation.add_scalar(f'metrics/cosine_similarity', get_cosine_similarity(validation_prediction, validation_label), iteration)
        writer.add_scalar(f'metrics/KL-divergence', get_kl_divergence(predicted_batch, label_batch), iteration)
        writer_validation.add_scalar(f'metrics/KL-divergence', get_kl_divergence(validation_prediction, validation_label), iteration)
        writer.add_scalar(f'metrics/JS-divergence', get_jensen_shannon(predicted_batch, label_batch), iteration)
        writer_validation.add_scalar(f'metrics/JS-divergence', get_jensen_shannon(validation_prediction, validation_label), iteration)
        writer.add_scalar(f'metrics/mse', get_MSE(predicted_batch, label_batch), iteration)
        writer_validation.add_scalar(f'metrics/mse', get_MSE(validation_prediction, validation_label), iteration)

        if iteration % 500 == 0:
            torch.save(sn.state_dict(), os.path.join("models", experiment_id))

    torch.save(sn.state_dict(), os.path.join("models", experiment_id))