import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.dataloader import DataLoader

torch.seed = 0
np.random.seed(0)

if __name__ == "__main__":
    num_classes = 30

    # Generate data
    data = pd.read_excel("../../data/data_v2.xls")
    print(data)
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                for i in range(1, 31)][:num_classes]
    dataloader = DataLoader(signatures=signatures,
                            batch_size=10000,
                            n_samples=0,    # random number of mutations
                            min_n_signatures=1,
                            max_n_signatures=10,
                            seed=0)
    #input_batch, label_batch = dataloader.get_batch(normalize=True)
    input_batch, label_batch = dataloader.make_random_set("train", normalize=True)
    df = input_batch.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv("../../data/exp_v2/train_random_input.csv", header=False, index=False)

    df = label_batch.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv("../../data/exp_v2/train_random_label.csv", header=False, index=False)

    input_batch, label_batch = dataloader.make_random_set("val", normalize=True)
    df = input_batch.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv("../../data/exp_v2/val_random_input.csv", header=False, index=False)

    df = label_batch.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv("../../data/exp_v2/val_random_label.csv", header=False, index=False)

    input_batch, label_batch = dataloader.make_random_set("test", normalize=True)
    df = input_batch.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv("../../data/exp_v2/test_random_input.csv", header=False, index=False)

    df = label_batch.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv("../../data/exp_v2/test_random_label.csv", header=False, index=False)