import os
import logging
import pathlib

import pandas as pd
import torch

from signet import DATA, TRAINED_MODELS
from signet.utilities.normalize_data import normalize_data
from signet.utilities.io import read_model, read_signatures
from signet.models import Baseline
from signet.modules import CombinedFinetuner, ClassifiedFinetunerErrorfinder
from utilities.plotting import plot_weights

class SigNet:

    def __init__(self,
                 classifier=os.path.join(TRAINED_MODELS, "detector"),
                 finetuner_realistic_low=os.path.join(TRAINED_MODELS, "finetuner_low"),
                 finetuner_realistic_large=os.path.join(TRAINED_MODELS, "finetuner_large"),
                 errorfinder=os.path.join(TRAINED_MODELS, "errorfinder"),
                 opportunities_name_or_path=None,
                 signatures_path=os.path.join(DATA, "data.xlsx"),
                 mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx")):

        signatures = read_signatures(file=signatures_path,
                                     mutation_type_order=mutation_type_order)
        self.signatures = signatures # TODO(oleguer): Remove, this is only for debugging
        self.baseline = Baseline(signatures)
        finetuner = CombinedFinetuner(low_mum_mut_dir=finetuner_realistic_low,
                                      large_mum_mut_dir=finetuner_realistic_large)

        classifier = read_model(classifier)
        errorfinder = read_model(errorfinder)

        self.finetuner_errorfinder = ClassifiedFinetunerErrorfinder(classifier=classifier,
                                                                    finetuner=finetuner,
                                                                    errorfinder=errorfinder)
        self.opportunities_name_or_path = opportunities_name_or_path\
            if opportunities_name_or_path != 'None' else None
        logging.info("SigNet loaded!")

    def __call__(self,
                 mutation_dataset,
                 numpy=True,
                 nworkers=1):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_dataset (pd.DataFrame(batch_size, 96)): Batch of mutation vectors to decompose
            numpy (bool): Whether to convert the outputs into numpy arrays (otherwise it'd be torch.Tensor). Default: True
            nworkers (int): Num of threads to run the

        Returns:
            results (dict)
        """
        with torch.no_grad():
            # Sort input data columns
            mutation_order = pd.read_excel(os.path.join(DATA, "mutation_type_order.xlsx"))
            mutation_dataset = mutation_dataset[list(mutation_order['Type'])]
            sample_names = mutation_dataset.index

            mutation_vec = torch.tensor(mutation_dataset.values, dtype=torch.float, device='cpu')

            # Normalize input data
            if self.opportunities_name_or_path is not None:
                mutation_vec = normalize_data(mutation_vec, self.opportunities_name_or_path)

            sums = torch.sum(mutation_vec, dim=1).reshape(-1, 1)
            normalized_mutation_vec = mutation_vec / sums
  
            # Run NNLS
            logging.info("Obtaining NNLS guesses...")
            self.baseline_guess = self.baseline.get_weights_batch(input_batch=normalized_mutation_vec, 
                                                                  n_workers=nworkers)
            logging.info("Obtaining NNLS guesses... DONE")


            # Finetune guess and aproximate errors
            num_mutations = torch.sum(mutation_vec, dim=1)
            signet_res = self.finetuner_errorfinder(mutation_dist=normalized_mutation_vec,
                                                    baseline_guess=self.baseline_guess,
                                                    num_mut=num_mutations.reshape(-1, 1))
            result = SigNetResult(mutation_dataset,
                                  weights=signet_res["finetuner_guess"],
                                  lower=signet_res["error_lower"],
                                  upper=signet_res["error_upper"],
                                  classification=signet_res["classification"],
                                  normalized_input=normalized_mutation_vec)
        return result

class SigNetResult:

    def __init__(self,
                 mutation_dataset,
                 weights,
                 lower,
                 upper,
                 classification,
                 normalized_input):
        self.mutation_dataset = mutation_dataset
        self.weights = weights
        self.lower = lower
        self.upper = upper
        self.classification = classification
        self.normalized_input = normalized_input
        self.sig_names = list(pd.read_excel(os.path.join(DATA, "data.xlsx")).columns)[1:]

    def convert_output(self, convert_to="numpy"):
        if convert_to == "numpy":
            weights = self.weights.detach().numpy()
            lower = self.lower.detach().numpy()
            upper = self.upper.detach().numpy()
            classification = self.classification.detach().numpy()
            normalized_input = self.normalized_input.detach().numpy()
        if convert_to == "pandas":
            weights = pd.DataFrame(self.weights.detach().numpy())
            lower = pd.DataFrame(self.lower.detach().numpy())
            upper = pd.DataFrame(self.upper.detach().numpy())
            classification = pd.DataFrame(self.classification.detach().numpy())
            normalized_input = pd.DataFrame(self.normalized_input.detach().numpy())
        if convert_to == "tensor":
            return self.weights, self.lower, self.upper, self.classification, self.normalized_input
        return weights, lower, upper, classification, normalized_input

    def save(self, 
             path='Output',
             name='run0'):
        logging.info("Writting results: %s"%path)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        weights, lower, upper, classification, _ = self.convert_output(convert_to="pandas")

        weights.columns = self.sig_names + ['Unknown']
        weights.index = self.mutation_dataset.index
        weights.to_csv(path + "/weight_guesses-%s.csv"%name, header=True, index=True)

        lower.columns = self.sig_names
        lower.index = self.mutation_dataset.index
        lower.to_csv(path + "/lower_bound_guesses-%s.csv"%name, header=True, index=True)
        
        upper.columns = self.sig_names
        upper.index = self.mutation_dataset.index
        upper.to_csv(path + "/upper_bound_guesses-%s.csv"%name, header=True, index=True)
        
        classification.columns = ['Classification']
        classification.index = self.mutation_dataset.index
        classification.to_csv(path + "/classification_guesses-%s.csv"%name, header=True, index=True)

    def plot_results(self, 
                     save = True,
                     path = 'Output/plots'):
        logging.info("Plotting results: %s"%path)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        samples = list(self.mutation_dataset.index)
        for i in range(self.weights.shape[0]):
            plot_weights(guessed_labels=self.weights[i,:72], 
                         pred_upper=self.upper[i,:], 
                         pred_lower=self.lower[i,:], 
                         sigs_names=self.sig_names, 
                         save=save, 
                         plot_path=path + "/plot_%s.png"%samples[i])