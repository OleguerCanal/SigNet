import os
import sys

#import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from error_finder import ErrorFinder
from finetuner import FineTuner
from signature_finder import SignatureFinder
from utilities.normalize_data import create_opportunities, normalize_data
from utilities.metrics import get_jensen_shannon
from utilities.plotting import plot_weights, plot_weights_comparison


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

        # Instantiate finetuner and read params
        self.finetuner = FineTuner(**finetuner_params)
        self.finetuner.load_state_dict(torch.load(os.path.join(
            models_path, finetuner_model_name), map_location=torch.device('cpu')))
        self.finetuner.eval()

        # Instantiate errorfinder and read params
        self.error_finder = ErrorFinder(**error_learner_params)
        self.error_finder.load_state_dict(torch.load(os.path.join(
            models_path, error_finder_model_name), map_location=torch.device('cpu')))
        self.error_finder.eval()

        self.opp = create_opportunities(path_opportunities)

    def __call__(self,
                 mutation_vec):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_vec (np.array(batch_size, 96)): Batch of mutation vectors to decompose
        """
        with torch.no_grad():
            # Normalize input data
            num_mutations = torch.sum(mutation_vec, dim=1)
            normalized_mutation_vec = normalize_data(mutation_vec, self.opp)
            normalized_mutation_vec = normalized_mutation_vec / torch.sum(normalized_mutation_vec, dim=1).reshape(-1,1)

            # Run signature_finder
            weight_guess_0 = self.signature_finder.get_weights_batch(
                normalized_mutation_vec)

            # Run finetuner
            weight_guess_1 = self.finetuner(mutation_dist=normalized_mutation_vec,
                                            weights=weight_guess_0)
            # Run error_finder
            positive_errors, negative_errors = self.error_finder(weights=weight_guess_1,
                                                                 num_mutations=num_mutations.reshape(-1,1))
        return weight_guess_0, weight_guess_1, positive_errors, negative_errors


if __name__ == "__main__":
    num_classes = 72
    data = pd.read_excel("../../data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]
    signature_finder_params = {"signatures": signatures,
                               "metric": get_jensen_shannon}

    finetuner_model_name = "finetuner_model_4"
    finetuner_params = {"num_hidden_layers": 1,
                        "num_units": 1500,
                        "num_classes": 72}

    error_finder_model_name = "error_finder_model_1"
    error_learner_params = {"num_hidden_layers_pos": 1,
                            "num_units_pos": 1500,
                            "num_hidden_layers_neg": 1,
                            "num_units_neg": 1500,
                            "normalize_mut": 2e4}

    path_opportunities = "../../data/data_donors/abundances_trinucleotides.txt"
    signature_net = SignatureNet(signature_finder_params, finetuner_params, error_learner_params,
                                path_opportunities, finetuner_model_name, error_finder_model_name)

    mutation_data = torch.tensor(pd.read_csv("../../data/data_donors/MC3_data/MC3_data_total.csv", header=None).values, dtype=torch.float)
    weight0, weight, pos, neg = signature_net(mutation_vec=mutation_data)

    deconstructSigs_batch = torch.tensor(pd.read_csv("../../data/data_donors/MC3_data/MC3_deconstructSigs.csv", header=None).values, dtype=torch.float)

    plot_weights_comparison( deconstructSigs_batch[0,:].detach().numpy(), weight[0,:].detach().numpy(), pos[0,:].detach().numpy(),neg[0,:].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison( deconstructSigs_batch[1,:].detach().numpy(), weight[1,:].detach().numpy(), pos[1,:].detach().numpy(),neg[1,:].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison( deconstructSigs_batch[2,:].detach().numpy(), weight[2,:].detach().numpy(), pos[2,:].detach().numpy(),neg[2,:].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison(deconstructSigs_batch[22,:].detach().numpy(),weight[22,:].detach().numpy(), pos[22,:].detach().numpy(),neg[22,:].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison(deconstructSigs_batch[3,:].detach().numpy(),weight[3,:].detach().numpy(), pos[3,:].detach().numpy(), neg[3,:].detach().numpy(), list(data.columns)[2:])
    plot_weights_comparison(deconstructSigs_batch[-10,:].detach().numpy(),weight[-10,:].detach().numpy(), pos[-10,:].detach().numpy(),neg[-10,:].detach().numpy(), list(data.columns)[2:])
    # plot_weights(weight[0, :].detach().numpy(), pos[0, :].detach().numpy(), neg[0, :].detach().numpy(), list(data.columns)[2:])
    # plot_weights(weight0[0, :].detach().numpy(),weight0[0, :].detach().numpy(), weight0[0, :].detach().numpy(), list(data.columns)[2:])
    # plot_weights(weight[1, :].detach().numpy(), pos[1, :].detach().numpy(), neg[1, :].detach().numpy(), list(data.columns)[2:])
    # plot_weights(weight[2, :].detach().numpy(), pos[2, :].detach().numpy(), neg[2, :].detach().numpy(), list(data.columns)[2:])
