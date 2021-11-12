import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_model

class CombinedFinetuner:
    def __init__(self,
                 low_mum_mut_dir,
                 large_mum_mut_dir):

        # Instantiate finetuner 1 and read params
        self.finetuner_low = read_model(low_mum_mut_dir)
        self.finetuner_large = read_model(large_mum_mut_dir)

    def __call__(self,
                 mutation_dist,
                 baseline_guess,
                 num_mut):
        """Get weights of each signature in lexicographic wrt 1-mer
        """
        num_mut = num_mut.view(-1)
        ind = torch.tensor(range(mutation_dist.size()[0]))
        ind_order = ind[num_mut <= 1e3]
        ind_order = torch.cat((ind_order, ind[num_mut > 1e3]))
        input_batch_low = mutation_dist[num_mut <= 1e3, ]
        input_batch_large = mutation_dist[num_mut > 1e3, ]
        baseline_guess_low = baseline_guess[num_mut <= 1e3, ]
        baseline_guess_large = baseline_guess[num_mut > 1e3, ]
        num_mut_low = num_mut[num_mut <= 1e3, ]
        num_mut_large = num_mut[num_mut > 1e3, ]
        num_mut_low = num_mut_low.reshape(-1, 1)
        num_mut_large = num_mut_large.reshape(-1, 1)
        with torch.no_grad():
            guess_low = self.finetuner_low(
                input_batch_low, baseline_guess_low, num_mut_low)
            guess_large = self.finetuner_large(
                input_batch_large, baseline_guess_large, num_mut_large)
            finetuner_guess = torch.cat((guess_low, guess_large), dim=0)
            ind_order = [float(el) for el in ind_order]
            finetuner_guess = torch.cat(
                (finetuner_guess, torch.tensor(ind_order).reshape(-1, 1)), dim=1)
            finetuner_guess = finetuner_guess[finetuner_guess[:, -1].sort()[1]]
            finetuner_guess = finetuner_guess[:, :-1]

        return finetuner_guess


def baseline_guess_to_combined_finetuner_guess(finetuner_low_dir, finetuner_large_dir, data):
    # Load finetuner and compute guess_1
    import gc

    combined_finetuner = CombinedFinetuner(low_mum_mut_dir=finetuner_low_dir,
                                           large_mum_mut_dir=finetuner_large_dir)

    with torch.no_grad():
        data.prev_guess = combined_finetuner(mutation_dist=data.inputs,
                                    baseline_guess=data.prev_guess,
                                    num_mut=data.num_mut)
    del combined_finetuner
    gc.collect()
    torch.cuda.empty_cache()
    return data