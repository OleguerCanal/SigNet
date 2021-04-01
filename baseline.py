from concurrent.futures import ProcessPoolExecutor
import os
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.metrics import *

class SignatureFinder:
    # TODO(oleguer): Use torch tensors instead of np arrays everywhere

    def __init__(self, signatures, metric=None, lagrange_mult=0.1):
        self.signatures = torch.stack(signatures).t().cpu().detach().numpy()
        self.__weight_len = self.signatures.shape[1]
        self.__bounds = [(0, 1)]*self.__weight_len
        self.lagrange_mult = lagrange_mult
        self.metric = metric if metric is not None else get_MSE
        # Define sum-1 constrain

        def __constrain(w):
            return np.sum(w) - 1
        self.constraints = [{"type": "eq", "fun": __constrain}]

    def __objective(self, w):
        with torch.no_grad():
            guess = np.dot(self.signatures, w)
            guess_tensor = torch.tensor(guess, dtype=torch.float).unsqueeze(0)
            error = self.metric(self.signature_tensor, guess_tensor).item()
        # return error + self.lagrange_mult*(np.sum(w) - 1)**2
        return error

    def get_weights(self, signature):
        w = np.random.uniform(low=0, high=1, size=(self.__weight_len,))
        self.signature_tensor = torch.tensor(signature, dtype=torch.float).unsqueeze(0)
        res = minimize(self.__objective, w,
                       bounds=self.__bounds,
                       constraints=self.constraints
                       )
        return res.x

    def get_weights_batch(self, input_batch):
        result = []
        # id_array = [*range(input_batch.shape[0])]
        input_as_list = input_batch.tolist()
        with ProcessPoolExecutor(max_workers=8) as executor:
            for r in executor.map(self.get_weights, input_as_list):
                result.append(torch.tensor(r, dtype=torch.float32))
        guessed_labels = torch.stack(result)
        return guessed_labels


if __name__ == "__main__":
    num_classes = 72
    training_data =  torch.tensor(pd.read_csv("data/validation_input.csv", header=None).values, dtype=torch.float)
    print(training_data)
    data = pd.read_excel("data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]
    sf = SignatureFinder(signatures, metric=get_jensen_shannon)
    # signature = 0.5*sf.signatures[:, 0] + 0.3*sf.signatures[:, 1] +\
    #     0.1*sf.signatures[:, 2] + 0.1*sf.signatures[:, 3]
    # sol = sf.get_weights(signature)
    # print(np.round(sol, decimals=3))
    # print(np.sum(sol))
    # sf = SignatureFinder(signatures)
    
    sol = sf.get_weights_batch(training_data)
    sol = sol.detach().numpy()
    df = pd.DataFrame(sol)
    df.to_csv("data/validation_baseline.csv", header=False, index=False)
