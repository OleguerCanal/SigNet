from concurrent.futures import ProcessPoolExecutor
import copy
import os
import sys
import gc

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.optimize import nnls
import torch
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.metrics import *


class YapsaInspiredBaseline:

    def __init__(self, signatures):
        self.signatures = torch.stack(signatures).t().cpu().detach().numpy()
        self.__weight_len = self.signatures.shape[1]

    def get_weights(self, normalized_mutations):
        h, rnorm = nnls(self.signatures, normalized_mutations,
                        maxiter=5*self.__weight_len)
        return torch.from_numpy(h).float()

    def get_weights_batch(self, input_batch, n_workers=4):
        result = []
        input_as_list = input_batch.tolist()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for r in executor.map(self.get_weights, input_as_list):
                result.append(copy.deepcopy(r))
                del r
        guessed_labels = torch.stack(result)
        return guessed_labels


def create_baseline_dataset(input_file, output_file):
    num_classes = 72
    data_path = "../../data/"

    signatures_data = pd.read_excel(data_path + "data.xlsx")
    signatures = [torch.tensor(signatures_data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]

    sf = YapsaInspiredBaseline(signatures)

    input_data = torch.tensor(pd.read_csv(
        data_path + input_file, header=None).values, dtype=torch.float)
    sol = sf.get_weights_batch(input_data)
    sol = sol.detach().numpy()

    print("done")


def create_huge_baseline_dataset(input_file, output_file):
    num_classes = 72
    data_path = "../../data/"

    signatures_data = pd.read_excel(data_path + "data.xlsx")
    signatures = [torch.tensor(signatures_data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]

    sf = YapsaInspiredBaseline(signatures)

    x = np.linspace(0, int(1e7), num=1000, dtype=int)
    for i in tqdm(range(len(x)-1)):
        input_data = torch.tensor(pd.read_csv(
            data_path + input_file, header=None, skiprows=x[i], nrows=x[i+1] - x[i]).values, dtype=torch.float)
        sol_i = sf.get_weights_batch(input_data)
        df = pd.DataFrame(sol_i)
        df.to_csv(data_path + output_file, header=False, index=False, mode='a')
        del sol_i
        gc.collect()
        
    print("Done!")


if __name__ == "__main__":
    training_data_in_file = "/random_train/train_random_input.csv"
    training_data_in_file_2 = "/realistic_train/train_realistic_input.csv"
    # validation_data_in_file = "validation_input_w01.csv"
    # test_data_in_file = "test_input_w01.csv"

    training_data_out_file = "/random_train/train_random_baseline_yapsa.csv"
    training_data_out_file_2 = "/realistic_train/train_realistic_baseline_yapsa.csv"
    # training_data_out_file = "train_baseline_yapsa_huge.csv"
    # validation_data_out_file = "validation_baseline_yapsa.csv"
    # test_data_out_file = "test_baseline_yapsa.csv"

    # create_baseline_dataset(training_data_in_file, training_data_out_file)
    create_baseline_dataset(training_data_in_file_2, training_data_out_file_2)
    # create_baseline_dataset(validation_data_in_file, validation_data_out_file)
    # create_baseline_dataset(test_data_in_file, test_data_out_file)
