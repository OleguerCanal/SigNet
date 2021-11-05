import os
import sys

#import numpy as np
import pandas as pd
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.classified_tunning_error import ClassifiedFinetunerErrorfinder
from modules.combined_finetuner import CombinedFinetuner
from modules.combined_errorfinder import CombinedErrorfinder
from models.baseline import Baseline
from utilities.io import read_model, read_signatures
from utilities.normalize_data import create_opportunities, normalize_data
from utilities.metrics import get_jensen_shannon
from utilities.plotting import plot_weights, plot_weights_comparison


class SigNet:
    def __init__(self,
                 classifier,
                 finetuner_random_low,
                 finetuner_random_large,
                 finetuner_realistic_low,
                 finetuner_realistic_large,
                 errorfinder_random_low,
                 errorfinder_random_large,
                 errorfinder_realistic_low,
                 errorfinder_realistic_large,
                 path_opportunities = None,
                 signatures_path = "../../data/data.xlsx"):

        self.signatures = read_signatures(signatures_path)

        self.classifier = classifier
        self.finetuner_random_low = finetuner_random_low
        self.finetuner_random_large= finetuner_random_large
        self.finetuner_realistic_low= finetuner_realistic_low
        self.finetuner_realistic_large= finetuner_realistic_large
        self.errorfinder_random_low= errorfinder_random_low
        self.errorfinder_random_large= errorfinder_random_large
        self.errorfinder_realistic_low= errorfinder_realistic_low
        self.errorfinder_realistic_large= errorfinder_realistic_large

        if path_opportunities != None:
            self.opp = create_opportunities(path_opportunities)
        else:
            self.opp = None

    def __call__(self,
                 mutation_vec):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_vec (np.array(batch_size, 96)): Batch of mutation vectors to decompose
        """
        with torch.no_grad():
            # Normalize input data
            num_mutations = torch.sum(mutation_vec, dim=1)

            if self.opp is not None:
                normalized_mutation_vec = normalize_data(mutation_vec, self.opp)
                normalized_mutation_vec = normalized_mutation_vec / torch.sum(normalized_mutation_vec, dim=1).reshape(-1,1)
            else:
                normalized_mutation_vec = mutation_vec / torch.sum(mutation_vec, dim=1).reshape(-1,1)

            # Run signature_finder
            baseline = Baseline(self.signatures)
            baseline_guess = baseline.get_weights_batch(normalized_mutation_vec, n_workers=1)

            realistic_finetuner = CombinedFinetuner(low_mum_mut_dir=self.finetuner_realistic_low,
                                                    large_mum_mut_dir=self.finetuner_realistic_large)

            random_finetuner = CombinedFinetuner(low_mum_mut_dir=self.finetuner_random_low,
                                                    large_mum_mut_dir=self.finetuner_random_large)

            realistic_errorfinder = CombinedErrorfinder(low_mum_mut_dir=self.errorfinder_realistic_low,
                                                    large_mum_mut_dir=self.errorfinder_realistic_large)

            random_errorfinder = CombinedErrorfinder(low_mum_mut_dir=self.errorfinder_random_low,
                                                    large_mum_mut_dir=self.errorfinder_random_large)

            finetuner_errorfinder = ClassifiedFinetunerErrorfinder(classifier=read_model(self.classifier),
                                            realistic_finetuner=realistic_finetuner,
                                            random_finetuner=random_finetuner,
                                            realistic_errorfinder=realistic_errorfinder,
                                            random_errorfinder=random_errorfinder)

            finetuner_guess, upper_bound, lower_bound = finetuner_errorfinder(normalized_mutation_vec, baseline_guess, num_mutations.reshape(-1,1))

        return finetuner_guess, upper_bound, lower_bound


if __name__ == "__main__":

    model_path = "../../trained_models/"
    experiment_id = "exp_final"

    # Model ids
    classifier = model_path + "classifier"
    finetuner_realistic_low = model_path + experiment_id + "/finetuner_realistic_low"
    finetuner_realistic_large = model_path + experiment_id + "/finetuner_realistic_large"
    finetuner_random_low = model_path + experiment_id + "/finetuner_realistic_low"
    finetuner_random_large = model_path + experiment_id + "/finetuner_realistic_large"
    errorfinder_random_low = model_path + experiment_id + "/errorfinder_realistic_low"
    errorfinder_random_large = model_path + experiment_id + "/errorfinder_realistic_large"
    errorfinder_realistic_low = model_path + experiment_id + "/errorfinder_realistic_low"
    errorfinder_realistic_large = model_path + experiment_id + "/errorfinder_realistic_large"

    # Things with real data:
    path_opportunities = "../../data/data_donors/abundances_trinucleotides.txt"
    signet = SigNet(classifier,
                    finetuner_random_low,
                    finetuner_random_large,
                    finetuner_realistic_low,
                    finetuner_realistic_large,
                    errorfinder_random_low,
                    errorfinder_random_large,
                    errorfinder_realistic_low,
                    errorfinder_realistic_large,
                    path_opportunities = path_opportunities,
                    signatures_path = "../../data/data.xlsx")

    # mutation_data = torch.tensor(pd.read_csv("../../data/case_study/data_by_tissue/all_tissues_input.csv", header=None).values, dtype=torch.float)
    mutation_data = torch.tensor(pd.read_csv("../../data/case_study/data_by_donor/all_donors_input.csv", header=None).values, dtype=torch.float)
    allfiles = [f for f in os.listdir("../../data/case_study/data_by_donor")]
    weight_guess, upper_bound, lower_bound = signet(mutation_vec=mutation_data)

    tissues = ["AdiposeTissue","AdrenalGland","Bladder","BloodVessel","Brain","Breast","CervixUteri","Colon","Esophagus","Heart","Kidney",
                    "Liver","Lung","Muscle","Nerve","Ovary","Pancreas","Pituitary","Prostate","SalivaryGland","Skin","SmallIntestine",
                    "Spleen","Stomach","Testis","Thyroid","Uterus","Vagina"]
    for i in range(mutation_data.size(0)):
        plot_weights(weight_guess[i, :].detach().numpy(), upper_bound[i, :].detach().numpy(), lower_bound[i, :].detach().numpy(), list(pd.read_excel("../../data/data.xlsx").columns)[1:], '../../plots/case_study/skin/%s.png'%allfiles[i])

    # deconstructSigs_batch = torch.tensor(pd.read_csv("../../data/data_donors/MC3_ACC_deconstructSigs.csv", header=None).values, dtype=torch.float)
    # plot_weights_comparison(deconstructSigs_batch[0,:].detach().numpy(), weight_guess[0,:].detach().numpy(), upper_bound[0,:].detach().numpy(), lower_bound[0,:].detach().numpy(), list(pd.read_excel("../../data/data.xlsx").columns)[1:], '')
    
    # df = weight_guess.detach().numpy()
    # df = pd.DataFrame(df)
    # df.to_csv("../../data/realistic_data/methods/realistic_trained_signatures-net_realistic_test_guess.csv", header=False, index=False)
