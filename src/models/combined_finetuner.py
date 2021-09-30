import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.finetuner import FineTuner

class CombinedFinetuner:
    def __init__(self,
                 experiment_id,
                 finetuner_model_name_low,
                 finetuner_params_low,
                 finetuner_model_name_large,
                 finetuner_params_large,
                 models_path="../../trained_models"):

        # Instantiate finetuner 1 and read params
        self.finetuner_low = FineTuner(**finetuner_params_low)
        self.finetuner_low.load_state_dict(torch.load(os.path.join(
            models_path, experiment_id, finetuner_model_name_low), map_location=torch.device('cpu')))
        self.finetuner_low.eval()  #NOTE: Careful! Only for evaluation (train submodels individually)

        # Instantiate finetuner 2 and read params
        self.finetuner_large = FineTuner(**finetuner_params_large)
        self.finetuner_large.load_state_dict(torch.load(os.path.join(
            models_path, experiment_id, finetuner_model_name_large), map_location=torch.device('cpu')))
        self.finetuner_large.eval()  #NOTE: Careful! Only for evaluation (train submodels individually)

    def __call__(self,
                 input_batch,
                 baseline_guess,
                 num_mut):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_vec (np.array(batch_size, 96)): Batch of mutation vectors to decompose
        """
        ind = np.array(range(input_batch.size()[0]))
        ind_order = ind[num_mut<1e3]
        ind_order = np.concatenate((ind_order, ind[num_mut>=1e3]))
        input_batch_low = input_batch[num_mut<1e3,]
        input_batch_large = input_batch[num_mut>=1e3,]
        baseline_guess_low = baseline_guess[num_mut<1e3,]
        baseline_guess_large = baseline_guess[num_mut>=1e3,]
        num_mut_low = num_mut[num_mut<1e3,]
        num_mut_large = num_mut[num_mut>=1e3,]
        num_mut_low = num_mut_low.reshape(-1,1)
        num_mut_large = num_mut_large.reshape(-1,1)
        with torch.no_grad():
            guess_low = self.finetuner_low(input_batch_low, baseline_guess_low, num_mut_low)
            guess_large = self.finetuner_large(input_batch_large, baseline_guess_large, num_mut_large)
            finetuner_guess = torch.cat((guess_low, guess_large), dim=0)
            finetuner_guess = torch.cat((finetuner_guess, torch.tensor(ind_order).reshape(-1,1)), dim=1)
            finetuner_guess = finetuner_guess[finetuner_guess[:, -1].sort()[1]]
            finetuner_guess = finetuner_guess[:,:-1]
        
        return finetuner_guess
            