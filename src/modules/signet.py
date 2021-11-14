import os
import sys

#import numpy as np
from argparse import ArgumentParser
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.plotting import plot_interval_performance, plot_weights
from utilities.normalize_data import normalize_data
from utilities.io import create_dir, read_model, read_signatures, update_dict
from models.baseline import Baseline
from modules.combined_errorfinder import CombinedErrorfinder
from modules.combined_finetuner import CombinedFinetuner
from modules.classified_tunning_error import ClassifiedFinetunerErrorfinder

class SigNet:
    def __init__(self,
                 classifier="../../trained_models/classifier",
                 finetuner_random_low="../../trained_models/exp_final/finetuner_random_low",
                 finetuner_random_large="../../trained_models/exp_final/finetuner_random_large",
                 finetuner_realistic_low="../../trained_models/exp_final/finetuner_realistic_low",
                 finetuner_realistic_large="../../trained_models/exp_final/finetuner_realistic_large",
                 errorfinder_random_low="../../trained_models/exp_final/errorfinder_random_low",
                 errorfinder_random_large="../../trained_models/exp_final/errorfinder_random_large",
                 errorfinder_realistic_low="../../trained_models/exp_final/errorfinder_realistic_low",
                 errorfinder_realistic_large="../../trained_models/exp_final/errorfinder_realistic_large",
                 opportunities_name_or_path=None,
                 signatures_path="../../data/data.xlsx",
                 mutation_type_order="../../data/mutation_type_order.xlsx"):

        signatures = read_signatures(file=signatures_path,
                                     mutation_type_order=mutation_type_order)
        self.baseline = Baseline(signatures)

        realistic_finetuner = CombinedFinetuner(low_mum_mut_dir=finetuner_realistic_low,
                                                large_mum_mut_dir=finetuner_realistic_large)

        random_finetuner = CombinedFinetuner(low_mum_mut_dir=finetuner_random_low,
                                             large_mum_mut_dir=finetuner_random_large)


        realistic_errorfinder = CombinedErrorfinder(low_mum_mut_dir=errorfinder_realistic_low,
                                                    large_mum_mut_dir=errorfinder_realistic_large)

        random_errorfinder = CombinedErrorfinder(low_mum_mut_dir=errorfinder_random_low,
                                                 large_mum_mut_dir=errorfinder_random_large)

        self.finetuner_errorfinder = ClassifiedFinetunerErrorfinder(classifier=read_model(classifier),
                                                                    realistic_finetuner=realistic_finetuner,
                                                                    random_finetuner=random_finetuner,
                                                                    realistic_errorfinder=realistic_errorfinder,
                                                                    random_errorfinder=random_errorfinder)
        self.opportunities_name_or_path = opportunities_name_or_path
        

    def __call__(self,
                 mutation_vec,
                 numpy=True,
                 nworkers=1):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_vec (np.array(batch_size, 96)): Batch of mutation vectors to decompose
        """
        with torch.no_grad():
            # Normalize input data
            num_mutations = torch.sum(mutation_vec, dim=1)

            if self.opportunities_name_or_path is not None:
                mutation_vec = normalize_data(mutation_vec, self.opportunities_name_or_path)

            normalized_mutation_vec = \
                mutation_vec / torch.sum(mutation_vec, dim=1).reshape(-1, 1)
  
            # Run signature_finder
            self.baseline_guess = self.baseline.get_weights_batch(
                normalized_mutation_vec, n_workers=nworkers)  # hack to be able to access it for benchmarking purposes

            finetuner_guess, upper_bound, lower_bound = self.finetuner_errorfinder(
                normalized_mutation_vec, self.baseline_guess, num_mutations.reshape(-1, 1))

        if numpy:
            return finetuner_guess.detach().numpy(), upper_bound.detach().numpy(), lower_bound.detach().numpy()
        return finetuner_guess, upper_bound, lower_bound


if __name__ == "__main__":
  
    config = {"input_data": None,
              "normalization": "exome",
              "output": "Output",
              "figures": "False"}

    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--input_data', action='store', nargs=1, type=str, required=False,
        help=f'Path to the input data to be analyzed.'
    )

    parser.add_argument(
        '--normalization', action='store', nargs=1, type=str, required=False, default=[None],
        help=f'The kind of normalization to be applied to the data. Should be either "None" (default), "exome", "genome" or a path to a file with the oppportunities.'
    )
    
    parser.add_argument(
        '--output', action='store', nargs=1, type=str, required=False, default=["Output"],
        help=f'Path to folder where all the output files will be saved. Default: "Output".'
    )

    parser.add_argument(
        '--figures', action='store', nargs=1, type=bool, required=False, default=[False],
        help=f'Boolean. Whether to compute plots for the output. Default: "False".'
    )
    
    _args = parser.parse_args()

    # Read & config
    config = update_dict(config=config, args=_args)
    print(config)

    input_file_path = config["input_data"]
    opportunities = config["normalization"]
    output_path = config["output"] 
    plot_figs = config["figures"]

    signet = SigNet(opportunities_name_or_path=opportunities, signatures_path="../../data/data.xlsx")

    mutation_data = torch.tensor(pd.read_csv(input_file_path, header=None, usecols=list(range(1,97))).values, dtype=torch.float)
    weight_guess, upper_bound, lower_bound = signet(mutation_vec=mutation_data)

    # Write results
    create_dir(output_path+ "/whatever.txt")
    df = pd.DataFrame(weight_guess)
    df.to_csv(output_path + "/all_donors_guesses.csv", header=False, index=False)


    # Plot figures
    if plot_figs:
        for i in range(weight_guess.shape[0]):
            plot_weights(weight_guess[i,:], upper_bound[i,:], lower_bound[i,:], list(pd.read_excel("../../data/data.xlsx").columns)[1:], output_path + "/plot_sample_%s.png"%str(i))