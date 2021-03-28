import os
import pathlib

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SignatureNet
from utilities.dataloader import DataLoader
from utilities.metrics import *
from utilities.plotting import plot_signature, plot_confusion_matrix

# Model params
num_hidden_layers = 4
num_neurons = 500
num_classes = 10

# Training params
experiment_id = "test_1"
iterations = 1e3
batch_size = 50
num_samples = 5000
intial_learning_rate = 0.01
learning_rate_steps = 20000
learning_rate_gamma = 0.1

if __name__ == "__main__":
    data = pd.read_excel("data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(
        torch.float32) for i in range(2, 74)][:num_classes]

    data_loader = DataLoader(signatures=signatures,
                             batch_size=batch_size,
                             n_samples=num_samples,
                             min_n_signatures=1,
                             max_n_signatures=5)

    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_id))

    sn = SignatureNet(num_classes=num_classes,
                      num_hidden_layers=num_hidden_layers, num_units=num_neurons)
    optimizer = optim.Adam(sn.parameters(), lr=intial_learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=learning_rate_steps, gamma=learning_rate_gamma)
    #loss = nn.CrossEntropyLoss()

    predicted_list = torch.zeros(0, dtype=torch.long)
    label_list = torch.zeros(0, dtype=torch.long)

    kl_div = torch.nn.KLDivLoss(reduction = 'batchmean')
    for iteration in tqdm(range(int(iterations))):
        input_batch, label_batch = data_loader.get_batch()
        optimizer.zero_grad()

        predicted_batch = sn(input_batch)

        l = get_cross_entropy(predicted_batch, label_batch)

        l.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar(f'metrics/cross-entropy', l.item(), iteration)
        writer.add_scalar(f'metrics/cosine_similarity', get_cosine_similarity(predicted_batch, label_batch), iteration)
        writer.add_scalar(f'metrics/KL-divergence', l.item() - get_entropy(label_batch), iteration)
        # writer.add_scalar(f'metrics/js-divergence', get_jensen_shannon(predicted_batch, label_batch), iteration)
        writer.add_scalar(f'metrics/mse', get_MSE(predicted_batch, label_batch), iteration)

    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(sn.state_dict(), os.path.join("models", experiment_id))
