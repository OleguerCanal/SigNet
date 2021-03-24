import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SignatureNet
from utilities import plot_signature, get_data_batches, get_entropy

# Model params
num_hidden_layers = 4
num_neurons = 500
num_classes = 4
learning_rate_gamma = 0.9

# Training params
experiment_id = "test_0"
iterations = 1e3
batch_size = 50

if __name__ == "__main__":
    data = pd.read_excel("data.xlsx")
    signatures = [torch.tensor(data.iloc[:,i]).type(torch.float32) for i in range(2, 74)]

    # TODO: Remove this line
    signatures = signatures[:num_classes]  # Classify only first 5 signatures

    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_id))

    sn = SignatureNet(num_classes=num_classes, num_hidden_layers=num_hidden_layers, num_units=num_neurons)
    optimizer = optim.Adam(sn.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=learning_rate_gamma)
    loss = nn.CrossEntropyLoss()

    for iteration in tqdm(range(int(iterations))):
        input_batch, label_batch = get_data_batches(signatures=signatures,
                                                    batch_size=batch_size)

        optimizer.zero_grad()

        predicted_batch = sn(input_batch)
        # print(predicted_batch)
        # print(label_batch)
        l = loss(predicted_batch, label_batch)

        writer.add_scalar(f'loss', l.item(), iteration)
        writer.add_scalar(f'entropy', get_entropy(predicted_batch), iteration)

        l.backward()
        optimizer.step()
        scheduler.step()

    torch.save(sn.state_dict(), os.path.join("models", experiment_id))

    for i in range(num_classes):
        prediction = sn(signatures[i].unsqueeze(dim=0))
        print(i)
        print(prediction)