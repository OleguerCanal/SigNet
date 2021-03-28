import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from tqdm import tqdm

class SignatureFinder:
    # TODO(oleguer): Use torch tensors instead of np arrays everywhere

    def __init__(self, signatures, lagrange_mult=0.1):
        self.signatures = torch.stack(signatures).t().cpu().detach().numpy()
        self.__weight_len = self.signatures.shape[1]
        self.__bounds = [(0, 1)]*self.__weight_len
        self.lagrange_mult = lagrange_mult

    def __objective(self, w, signature):
        return np.linalg.norm(signature - np.dot(self.signatures, w), ord=2) + self.lagrange_mult*(1 - np.sum(w))**2

    def get_weights(self, signature):
        w = np.random.uniform(low=0, high=1, size=(self.__weight_len,))
        res = minimize(self.__objective, w, args=(
            signature,), bounds=self.__bounds)
        return res.x

    def get_weights_batch(self, input_batch):
        guessed_labels = torch.empty((input_batch.shape[0], self.signatures.shape[1]))
        for i in tqdm(range(input_batch.shape[0])):
            guessed_labels[i] = torch.tensor(self.get_weights(input_batch[i, :]))
        return guessed_labels


if __name__ == "__main__":
    num_classes = 5
    data = pd.read_excel("data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]
    sf = SignatureFinder(data)
    signature = 0.5*sf.signatures[:, 0] + 0.3*sf.signatures[:,
                                                            1] + 0.1*sf.signatures[:, 2] + 0.1*sf.signatures[:, 3]
    sol = sf.get_weights(signature)
    print(np.round(sol, decimals=2))
    print(np.sum(sol))
