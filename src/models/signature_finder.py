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

    def __init__(self, signatures, metric=None, lagrange_mult=0.1):
        self.signatures = torch.stack(signatures).t().cpu().detach().numpy()
        self.__weight_len = self.signatures.shape[1]
        self.__bounds = [(0, 1)]*self.__weight_len
        self.lagrange_mult = lagrange_mult
        self.metric = metric if metric is not None else get_MSE

        # Define sum-1 constrain
        global constrain

        def constrain(w):
            return np.sum(w) - 1
        self.constraints = [{"type": "eq", "fun": constrain}]

    def __objective(self, w, normalized_mutations):
        with torch.no_grad():
            guess = np.dot(self.signatures, w)
            guess_tensor = torch.from_numpy(guess).unsqueeze(0)
            error = self.metric(guess_tensor, normalized_mutations)
        return error.numpy() + self.lagrange_mult*(np.sum(w) - 1)**2

    def get_weights(self, normalized_mutations):
        w = np.random.uniform(low=0, high=1, size=(self.__weight_len,))
        normalized_mutations = torch.from_numpy(np.array(normalized_mutations)).unsqueeze(0)
        res = minimize(fun=self.__objective,
                       x0=w,
                       args=(normalized_mutations,),
                       bounds=self.__bounds)
        return res.x

    def get_weights_batch(self, input_batch, n_workers=12):
        result = []
        input_as_list = input_batch.tolist()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for r in executor.map(self.get_weights, input_as_list):
                result.append(torch.tensor(r, dtype=torch.float32))
        guessed_labels = torch.stack(result)
        return guessed_labels


if __name__ == "__main__":
    num_classes = 72
    training_data = torch.tensor(pd.read_csv(
        "../../data/realistic_data/train_more_sigs/larger_realistic_train_input.csv", header=None).values, dtype=torch.float)
    # validation_data = torch.tensor(pd.read_csv(
    #     "../../data/realistic_data/train_default/realistic_validation_input.csv", header=None).values, dtype=torch.float)
    # test_data = torch.tensor(pd.read_csv(
        # "data/test_input.csv", header=None).values, dtype=torch.float)
    signatures_data = pd.read_excel("../../data/data.xlsx")
    signatures = [torch.tensor(signatures_data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]

    sf = SignatureFinder(signatures, metric=get_jensen_shannon)

    print("Computing...")
    sol = sf.get_weights_batch(training_data)
    sol = sol.detach().numpy()
    df = pd.DataFrame(sol)
    df.to_csv("../../data/realistic_data/train_more_sigs/larger_realistic_train_baseline_JS.csv", header=False, index=False)
    print("Training done!")

    # print("Computing...")
    # sol = sf.get_weights_batch(validation_data)
    # sol = sol.detach().numpy()
    # df = pd.DataFrame(sol)
    # df.to_csv("../../data/realistic_data/train_default/validation_baseline_JS.csv", header=False, index=False)
    # print("Validation done!")
    
    # sol = sf.get_weights_batch(test_data)
    # sol = sol.detach().numpy()
    # df = pd.DataFrame(sol)
    # df.to_csv("data/test_baseline_JS.csv", header=False, index=False)
    # print("Test done!")
