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
num_hidden_layers_pos = 2
num_neurons_pos = 500
num_hidden_layers_neg = 2
num_neurons_neg = 500
num_classes = 72

# Training params
experiment_id = "learn_error_posneg_large_4"
iterations = 4000
batch_size = 343
num_samples = 1500
intial_learning_rate = 0.001
learning_rate_steps = 10000
learning_rate_gamma = 1

if __name__ == "__main__":

    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    data = pd.read_excel("data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(
        torch.float32) for i in range(2, 74)][:num_classes]

    validation_input = torch.tensor(pd.read_csv("data/validation_input_w01.csv", header=None).values, dtype=torch.float)
    validation_mut_label = torch.tensor(pd.read_csv("data/validation_label_w01.csv", header=None).values, dtype=torch.float)
    validation_baseline = torch.tensor(pd.read_csv("data/validation_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    training_baseline = torch.tensor(pd.read_csv("data/train_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    training_input = torch.tensor(pd.read_csv("data/train_input_w01.csv", header=None).values, dtype=torch.float)
    training_label = torch.tensor(pd.read_csv("data/train_label_w01.csv", header=None).values, dtype=torch.float)
    
    data_loader = DataLoader(signatures=signatures,
                             batch_size=batch_size,
                             n_samples=num_samples,
                             min_n_signatures=1,
                             max_n_signatures=10)

    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_id))
    writer_validation = SummaryWriter(log_dir=os.path.join("runs/validation", experiment_id))

    sn = SignatureNet(signatures=signatures,
                      num_classes=num_classes,
                      num_hidden_layers_pos=num_hidden_layers_pos, num_units_pos=num_neurons_pos,
                      num_hidden_layers_neg=num_hidden_layers_neg, num_units_neg=num_neurons_neg)
    optimizer = optim.Adam(sn.parameters(), lr=intial_learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=learning_rate_steps, gamma=learning_rate_gamma)

    last_ind = 0
    for iteration in tqdm(range(int(iterations))):
        input_batch, label_mut_batch, baseline_batch, last_ind = data_loader.select_batch( training_input, training_label, training_baseline, last_ind)
        label_batch = label_mut_batch[:,:num_classes]
        num_mut = torch.reshape(label_mut_batch[:,num_classes], (batch_size,1))

        optimizer.zero_grad()

        predicted_error_pos, predicted_error_neg = sn(baseline_batch, num_mut)
        mt = ModelTester(num_classes)
        # l = get_cross_entropy(predicted_batch, label_batch)
        # l = get_MSE(predicted_batch, label_batch)
        # l = get_kl_divergence(predicted_batch, label_batch)
        # l = get_jensen_shannon(predicted_batch, label_batch)
        real_error = label_batch - baseline_batch
        real_error_pos = copy.deepcopy(real_error)
        real_error_pos[real_error < 0] = 0
        real_error_neg = copy.deepcopy(real_error)
        real_error_neg[real_error > 0] = 0
        l = get_MSE(predicted_error_pos, real_error_pos) + get_MSE(predicted_error_neg, real_error_neg)

        l.backward()
        optimizer.step()
        scheduler.step()

        validation_label = validation_mut_label[:,:num_classes]
        validation_mut =  torch.reshape(validation_mut_label[:,num_classes], (list(validation_label.size())[0],1))
        
        val_error_pos, val_error_neg = sn(validation_baseline, validation_mut)
        val_real_error = validation_label - validation_baseline
        val_real_error_pos = copy.deepcopy(val_real_error)
        val_real_error_pos[val_real_error < 0] = 0
        val_real_error_neg = copy.deepcopy(val_real_error)
        val_real_error_neg[val_real_error > 0] = 0
        
        writer.add_scalar(f'loss', l, iteration)
        writer_validation.add_scalar(f'loss',get_MSE(val_error_pos, val_real_error_pos) + get_MSE(val_error_neg, val_real_error_neg), iteration)
        if iteration % 500 == 0:
            torch.save(sn.state_dict(), os.path.join("models", experiment_id))
            print(predicted_error_pos)
            print(predicted_error_neg)

    torch.save(sn.state_dict(), os.path.join("models", experiment_id))