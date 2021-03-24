import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SignatureNet
from utilities import plot_signature, get_data_batches

experiment_id = "test_2"
iterations = 1e5
batch_size = 1000

if __name__ == "__main__":
    data = pd.read_excel("data.xlsx")
    signatures = [torch.tensor(data.iloc[:,i]).type(torch.float32) for i in range(2, 74)]

    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_id))

    sn = SignatureNet()
    optimizer = optim.Adam(sn.parameters(), lr=0.01)
    loss = nn.MSELoss()

    for iteration in tqdm(range(int(iterations))):
        input_batch, label_batch = get_data_batches(signatures=signatures,
                                                    batch_size=batch_size)

        optimizer.zero_grad()

        predicted_batch = sn(input_batch)
        l = loss(predicted_batch, label_batch)

        writer.add_scalar(f'loss', l.item(), iteration)

        l.backward()
        optimizer.step()

    torch.save(sn.state_dict(), os.path.join("models", experiment_id))

    saved_sn = SignatureNet()
    saved_sn.load_state_dict(torch.load(os.path.join("models", experiment_id)))
    prediction = saved_sn(signatures[0].unsqueeze(dim=0))
    print(prediction)