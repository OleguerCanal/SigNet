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


class SlowBaseline:

    def __init__(self, signatures, metric=None, lagrange_mult=0.1):
        self.signatures = signatures.cpu().detach().numpy()
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

    def get_weights_batch(self, input_batch, n_workers=8):
        result = []
        input_as_list = input_batch.tolist()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for r in executor.map(self.get_weights, input_as_list):
                result.append(torch.tensor(r, dtype=torch.float32))
        guessed_labels = torch.stack(result)
        return guessed_labels