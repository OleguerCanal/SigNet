import os

import numpy as np
import pandas as pd
import torch

from utilities.dataloader import DataLoader

if __name__ == "__main__":
    num_classes = 72

    # Generate data
    data = pd.read_excel("data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                for i in range(2, 74)][:num_classes]
    dataloader = DataLoader(signatures=signatures,
                            batch_size=500,
                            n_samples=5000,
                            min_n_signatures=1,
                            max_n_signatures=20)
    input_batch, label_batch = dataloader.get_batch(normalize=False)
    df = input_batch.detach().numpy()
    df = np.array(df, dtype=int)
    df = pd.DataFrame(df)
    df.to_csv("test_set.csv", header=False, index=False)