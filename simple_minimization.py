import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utilities import plot_signature

class SignatureFinder:
    def __init__(self, data):
        self.signatures = np.array([data.iloc[:,i].to_numpy() for i in range(2, 74)]).transpose()
        self.__weight_len = self.signatures.shape[1]
        self.__bounds = [(0, 1)]*self.__weight_len

    def __objective(self, w, signature, lagrange_mult):
        return np.linalg.norm(signature - np.dot(self.signatures, w), ord=2) + lagrange_mult*(1 - np.sum(w))**2

    def get_weights(self, signature, lagrange_mult=0.1):
        w = np.random.uniform(low=0, high=1, size=(self.__weight_len,))
        res = minimize(self.__objective, w, args=(signature, lagrange_mult,), bounds=self.__bounds)
        return res.x

if __name__=="__main__":
    data = pd.read_excel("data.xlsx")
    sf = SignatureFinder(data)
    signature = 0.5*sf.signatures[:, 0] + 0.3*sf.signatures[:, 3] + 0.1*sf.signatures[:, 4] + 0.1*sf.signatures[:, -3]
    # noise = 0.01*np.random.uniform(low=-1, high=1, size=(signature.shape[0]))
    # signature = np.abs(signature + noise)
    # plot_signature(signature)
    sol = sf.get_weights(signature)
    print(np.round(sol, decimals=2))
    print(np.sum(sol))