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

from models.slow_baseline import SlowBaseline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.metrics import get_jensen_shannon
from utilities.io import create_dir, read_signatures

class Baseline:

    def __init__(self, signatures):
        self.signatures = signatures.cpu().detach().numpy()
        self.__weight_len = self.signatures.shape[1]

    def get_weights(self, normalized_mutations):
        h, rnorm = nnls(self.signatures, normalized_mutations,
                        maxiter=5*self.__weight_len)
        return torch.from_numpy(h).float()

    def get_weights_batch(self, input_batch, n_workers=8):
        result = []
        input_as_list = input_batch.tolist()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for r in executor.map(self.get_weights, input_as_list):
                result.append(copy.deepcopy(r))
                del r
        guessed_labels = torch.stack(result)
        return guessed_labels

def create_baseline_dataset(input_file, output_file, signatures_path, which_baseline = "nnls"):
    signatures = read_signatures(signatures_path)

    if which_baseline == "nnls":
        sf = Baseline(signatures)
    else:
        sf = SlowBaseline(signatures, metric=get_jensen_shannon)

    input_data = torch.tensor(pd.read_csv(
        input_file, header=None).values, dtype=torch.float)
    sol = sf.get_weights_batch(input_data)
    sol = sol.detach().numpy()

    df = pd.DataFrame(sol)
    create_dir(output_file)
    df.to_csv(output_file, header=False, index=False)
    print("Baseline Done!")

if __name__ == "__main__":
    training_data_in_file = "/exp_v2/train_random_input.csv"
    validation_data_in_file = "/exp_v2/val_random_input.csv"
    test_data_in_file = "/exp_v2/test/test_random_input.csv"

    training_data_out_file = "/exp_v2/train_random_baseline.csv"
    validation_data_out_file = "/exp_v2/val_random_baseline.csv"
    test_data_out_file = "/exp_v2/test/test_random_baseline.csv"

    create_baseline_dataset(training_data_in_file, training_data_out_file)
    create_baseline_dataset(validation_data_in_file, validation_data_out_file)
    create_baseline_dataset(test_data_in_file, test_data_out_file)
