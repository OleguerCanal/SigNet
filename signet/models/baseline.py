
from concurrent.futures import ProcessPoolExecutor
import copy
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.optimize import nnls
import torch
from tqdm import tqdm

from signet import DATA
from signet.utilities.metrics import get_jensen_shannon, get_MSE
from signet.utilities.io import create_dir, read_signatures


class Baseline:

    def __init__(self, signatures, approximate_solution=True):
        """
        Args:
            signatures (pd.DataFrame): Signature catalog.
            approximate_solution (bool): Whether to use the approximate solution (less accurate but significantly faster)
        """
        self.signatures = signatures.cpu().detach().numpy()
        self.approximate_solution = approximate_solution
        self.lagrange_mult = 0.1
        self.__weight_len = self.signatures.shape[1]
        self.__bounds = [(0, 1)]*self.__weight_len

    def __precise_objective(self, w, normalized_mutations):
        with torch.no_grad():
            guess = np.dot(self.signatures, w)
            guess_tensor = torch.from_numpy(guess).unsqueeze(0)
            error = get_MSE(guess_tensor, normalized_mutations)
        return error.numpy() + self.lagrange_mult*(np.sum(w) - 1)**2

    def get_weights(self, normalized_mutations):
        if self.approximate_solution:
            h, rnorm = nnls(self.signatures, normalized_mutations,
                            maxiter=5*self.__weight_len)
            return torch.from_numpy(h).float()

        w = np.random.uniform(low=0, high=1, size=(self.__weight_len,))
        normalized_mutations = torch.from_numpy(np.array(normalized_mutations)).unsqueeze(0)
        res = minimize(fun=self.__precise_objective,
                        x0=w,
                        args=(normalized_mutations,),
                        bounds=self.__bounds)
        return torch.from_numpy(res.x).float()

    def get_weights_batch(self, input_batch, n_workers=8):
        result = []
        input_as_list = input_batch.tolist()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for r in executor.map(self.get_weights, input_as_list):
                result.append(copy.deepcopy(r))
                del r
        guessed_labels = torch.stack(result)
        return guessed_labels


def create_baseline_dataset(input_file, output_file, signatures_path, which_baseline="nnls"):
    signatures = read_signatures(signatures_path)

    if which_baseline == "nnls":
        sf = Baseline(signatures)
    else:
        raise NotImplementedError("Other optimization methods have been deprecated")

    input_data = torch.tensor(pd.read_csv(
        input_file, header=None).values, dtype=torch.float)
    sol = sf.get_weights_batch(input_data)
    sol = sol.detach().numpy()

    df = pd.DataFrame(sol)
    create_dir(output_file)
    df.to_csv(output_file, header=False, index=False)
    print("Baseline Done!")


if __name__ == "__main__":

    cosmic_version = str(sys.argv[1])

    if cosmic_version == 'v3':
        signatures_file = DATA + "data.xlsx"
        experiment_id = "exp_not_norm"
    elif cosmic_version == 'v2':
        signatures_file = DATA + "data_v2.xlsx"
        experiment_id = "exp_generator_v2"
    else:
        raise NotImplementedError

    in_files = [
        # generator
        "/%s/train_generator_low_input.csv"%experiment_id,
        "/%s/train_generator_large_input.csv"%experiment_id,
        "/%s/val_generator_low_input.csv"%experiment_id,
        "/%s/val_generator_large_input.csv"%experiment_id,
        "/%s/train_perturbed_low_input.csv"%experiment_id,
        "/%s/train_perturbed_large_input.csv"%experiment_id,
        "/%s/val_perturbed_low_input.csv"%experiment_id,
        "/%s/val_perturbed_large_input.csv"%experiment_id,

        # tests
        "/%s/test_generator_input.csv"%experiment_id,
        "/%s/test_perturbed_input.csv"%experiment_id,
    ]    

    for in_file in in_files:
        print("Computing baseline of:", in_file)
        try:
            in_file = data_folder + in_file
            out_file = in_file.replace("input", "baseline")
            create_baseline_dataset(input_file=in_file,
                                    output_file=out_file,
                                    signatures_path=signatures_file,
                                    which_baseline="nnls")
        except Exception as e:
            print(e)