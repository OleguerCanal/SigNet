import os
import sys

from numpy import np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from error_finder import ErrorFinder
from finetuner import FineTuner
from signature_finder import SignatureFinder
from utilities.normalize_data import create_opportunities, normalize_data


class SignatureNet:
    def __init__(self,
                 signature_finder_params,
                 finetuner_params,
                 error_learner_params,
                 path_opportunities,
                 finetuner_model_name,
                 error_finder_model_name,
                 models_path="../../trained_models"):
        self.signature_finder = SignatureFinder(**signature_finder_params)
        self.finetuner = FineTuner(**finetuner_params)
        self.error_finder = ErrorFinder(**error_learner_params)
        self.opp = create_opportunities(path_opportunities)
        self.finetuner_model_name = finetuner_model_name
        self.error_finder_model_name = error_finder_model_name
        self.models_path = models_path

    def __call__(self, 
                mutation_vec):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_vec (np.array(batch_size, 96)): Batch of mutation vectors to decompose
        """
        # Normalize input data
        num_mutations = np.sum(mutation_vec, axis=1)
        normalized_mutation_vec = normalize_data(mutation_vec, self.opp) 
        normalized_mutation_vec = normalized_mutation_vec/np.sum(normalized_mutation_vec, axis=1)

        # Run signature_finder
        weight_guess_0 = self.signature_finder.get_weights_batch(
            normalized_mutation_vec)

        # Run finetuner
        self.finetuner.load_state_dict(torch.load(os.path.join(self.models_path, self.finetuner_model_name), map_location=torch.device('cpu')))
        self.finetuner.eval()
        weight_guess_1 = self.finetuner(mutation_dist=normalized_mutation_vec,
                                           weights=weight_guess_0)

        # Run error_finder
        self.error_finder.load_state_dict(torch.load(os.path.join(self.models_path, self.error_finder_model_name), map_location=torch.device('cpu')))
        self.error_finder.eval()
        positive_errors, negative_errors = self.error_finder(weights=weight_guess_1,
                                                             num_mutations=num_mutations)
