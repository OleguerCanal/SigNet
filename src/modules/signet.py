import os
import sys

#import numpy as np
import pandas as pd
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.plotting import plot_weights
from utilities.normalize_data import create_opportunities, normalize_data
from utilities.io import read_model, read_signatures
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
                 errorfinder_random_low="../../trained_models/exp_final/errofinder_random_low",
                 errorfinder_random_large="../../trained_models/exp_final/errofinder_random_large",
                 errorfinder_realistic_low="../../trained_models/exp_final/errofinder_realistic_low",
                 errorfinder_realistic_large="../../trained_models/exp_final/errofinder_realistic_large",
                 path_opportunities=None,
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

        if path_opportunities != None:
            self.opp = create_opportunities(path_opportunities)
        else:
            self.opp = None

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

            if self.opp is not None:
                normalized_mutation_vec = normalize_data(
                    mutation_vec, self.opp)
                normalized_mutation_vec = normalized_mutation_vec / \
                    torch.sum(normalized_mutation_vec, dim=1).reshape(-1, 1)
            else:
                normalized_mutation_vec = mutation_vec / \
                    torch.sum(mutation_vec, dim=1).reshape(-1, 1)

            # Run signature_finder
            self.baseline_guess = self.baseline.get_weights_batch(
                normalized_mutation_vec, n_workers=nworkers)  # hack to be able to access it for benchmarking purposes

            finetuner_guess, upper_bound, lower_bound = self.finetuner_errorfinder(
                normalized_mutation_vec, self.baseline_guess, num_mutations.reshape(-1, 1))

        if numpy:
            return finetuner_guess.detach().numpy(), upper_bound.detach().numpy(), lower_bound.detach().numpy()
        return finetuner_guess, upper_bound, lower_bound


if __name__ == "__main__":

    # Things with real data:
    path_opportunities = "../../data/data_donors/abundances_trinucleotides.txt"
    signet = SigNet(path_opportunities=path_opportunities,
                    signatures_path="../../data/data.xlsx")

    # mutation_data = torch.tensor(pd.read_csv("../../data/case_study/data_by_tissue/all_tissues_input.csv", header=None).values, dtype=torch.float)
    mutation_data = torch.tensor(pd.read_csv(
        "../../data/case_study/data_by_donor/all_donors_input.csv", header=None).values, dtype=torch.float)
    allfiles = [f for f in os.listdir("../../data/case_study/data_by_donor")]
    weight_guess, upper_bound, lower_bound = signet(mutation_vec=mutation_data)

    tissues = ["AdiposeTissue", "AdrenalGland", "Bladder", "BloodVessel", "Brain", "Breast", "CervixUteri", "Colon", "Esophagus", "Heart", "Kidney",
               "Liver", "Lung", "Muscle", "Nerve", "Ovary", "Pancreas", "Pituitary", "Prostate", "SalivaryGland", "Skin", "SmallIntestine",
               "Spleen", "Stomach", "Testis", "Thyroid", "Uterus", "Vagina"]
    for i in range(mutation_data.size(0)):
        plot_weights(weight_guess[i, :], upper_bound[i, :], lower_bound[i, :], list(pd.read_excel(
            "../../data/data.xlsx").columns)[1:], '../../plots/case_study/skin/%s.png' % allfiles[i])

    # deconstructSigs_batch = torch.tensor(pd.read_csv("../../data/data_donors/MC3_ACC_deconstructSigs.csv", header=None).values, dtype=torch.float)
    # plot_weights_comparison(deconstructSigs_batch[0,:], weight_guess[0,:], upper_bound[0,:], lower_bound[0,:], list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')

    # df = weight_guess
    # df = pd.DataFrame(df)
    # df.to_csv("../../data/realistic_data/methods/realistic_trained_signatures-net_realistic_test_guess.csv", header=False, index=False)
