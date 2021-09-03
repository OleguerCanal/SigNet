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
    num_classes = 72

    # Generate data
    data = pd.read_excel("../../data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                for i in range(2, 74)][:num_classes]
    dataloader = DataLoader(signatures=signatures,
                            batch_size=10000,
                            n_samples=0,    # random number of mutations
                            min_n_signatures=1,
                            max_n_signatures=10,
                            seed=0)
    #input_batch, label_batch = dataloader.get_batch(normalize=True)
    input_batch, label_batch = dataloader.make_random_set("train", normalize=True)
    df = input_batch.detach().numpy()
    #df = np.array(df, dtype=int)
    df = pd.DataFrame(df)
    df.to_csv("../../data/train_val_test_sets/train_random_input.csv", header=False, index=False)

    df = label_batch.detach().numpy()
    df = pd.DataFrame(df)
    df.to_csv("../../data/train_val_test_sets/train_random_label.csv", header=False, index=False)