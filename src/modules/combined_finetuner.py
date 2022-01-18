import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_model
from utilities.metrics import get_reconstruction_error
from models.finetuner import FineTunerLowNumMut, FineTunerLargeNumMut

class CombinedFinetuner:
    def __init__(self,
                 low_mum_mut_dir,
                 large_mum_mut_dir):

        # Instantiate finetuner 1 and read params
        self.finetuner_low = read_model(low_mum_mut_dir)
        self.finetuner_large = read_model(large_mum_mut_dir)
        assert(isinstance(self.finetuner_low, FineTunerLowNumMut))  # finetuner_low model should be FineTunerLowNumMut
        assert(isinstance(self.finetuner_large, FineTunerLargeNumMut))  # finetuner_large model should be FineTunerLargeNumMut

    def __join_and_sort(self, low, large, ind_order):
        joined = torch.cat((low, large), dim=0)
        joined = torch.cat((joined, ind_order), dim=1)
        joined = joined[joined[:, -1].sort()[1]]
        return joined[:, :-1]
    
    def __call__(self,
                 mutation_dist,
                 baseline_guess,
                 num_mut):
        """Get weights of each signature in lexicographic wrt 1-mer
        """
        num_mut = num_mut.view(-1)
        ind = torch.tensor(range(mutation_dist.size()[0]))
        ind_order = torch.tensor(np.concatenate((ind[num_mut <= 1e3], ind[num_mut > 1e3]))).reshape(-1, 1).to(torch.float)
        
        input_batch_low = mutation_dist[num_mut <= 1e3, ]
        input_batch_large = mutation_dist[num_mut > 1e3, ]

        baseline_guess_low = baseline_guess[num_mut <= 1e3, ]
        baseline_guess_large = baseline_guess[num_mut > 1e3, ]
        
        num_mut_low = num_mut[num_mut <= 1e3, ].reshape(-1, 1)
        num_mut_large = num_mut[num_mut > 1e3, ].reshape(-1, 1)
        
        with torch.no_grad():
            guess_low = self.finetuner_low(
                input_batch_low, baseline_guess_low, num_mut_low)

            guess_large = self.finetuner_large(
                input_batch_large, baseline_guess_large, num_mut_large)
            
            finetuner_guess = self.__join_and_sort(low=guess_low,
                                                   large=guess_large,
                                                   ind_order=ind_order)
        return finetuner_guess


def baseline_guess_to_combined_finetuner_guess(model, data):
    # Load finetuner and compute guess_1
    import gc
    with torch.no_grad():
        data.prev_guess, data.classification = model(mutation_dist=data.inputs,
                                                     weights=data.prev_guess,
                                                     num_mut=data.num_mut)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return data