import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_model

class CombinedErrorfinder:
    def __init__(self,
                 low_mum_mut_dir,
                 large_mum_mut_dir):

        # Instantiate errorfinder 1 and read params
        self.errorfinder_low = read_model(low_mum_mut_dir)
        self.errorfinder_large = read_model(large_mum_mut_dir)

    def __call__(self,
                 finetuner_guess,
                 num_mut):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_vec (np.array(batch_size, 96)): Batch of mutation vectors to decompose
        """
        num_mut = num_mut.view(-1)
        ind = np.array(range(finetuner_guess.size()[0]))
        ind_order = ind[num_mut <= 1e3]
        ind_order = np.concatenate((ind_order, ind[num_mut > 1e3]))
        finetuner_guess_low = finetuner_guess[num_mut <= 1e3, ]
        finetuner_guess_large = finetuner_guess[num_mut > 1e3, ]
        num_mut_low = num_mut[num_mut <= 1e3, ]
        num_mut_large = num_mut[num_mut > 1e3, ]
        num_mut_low = num_mut_low.reshape(-1, 1)
        num_mut_large = num_mut_large.reshape(-1, 1)
        with torch.no_grad():
            upper_low, lower_low = self.errorfinder_low(finetuner_guess_low, num_mut_low)
            upper_large, lower_large = self.errorfinder_large(finetuner_guess_large, num_mut_large)
            upper_guess = torch.cat((upper_low, upper_large), dim=0)
            lower_guess = torch.cat((lower_low, lower_large), dim=0)
            ind_order = [float(el) for el in ind_order]
            upper_guess = torch.cat(
                (upper_guess, torch.tensor(ind_order).reshape(-1, 1)), dim=1)
            upper_guess = upper_guess[upper_guess[:, -1].sort()[1]]
            upper_guess = upper_guess[:, :-1]
            lower_guess = torch.cat(
                (lower_guess, torch.tensor(ind_order).reshape(-1, 1)), dim=1)
            lower_guess = lower_guess[lower_guess[:, -1].sort()[1]]
            lower_guess = lower_guess[:, :-1]

        return upper_guess, lower_guess