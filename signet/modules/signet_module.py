import os
import logging

import pandas as pd
import torch

from signet import DATA, TRAINED_MODELS
from signet.utilities.normalize_data import normalize_data
from signet.utilities.io import read_model, read_signatures
from signet.models import Baseline, ErrorFinder
from signet.modules import CombinedFinetuner, ClassifiedFinetunerErrorfinder

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
    
    def _convert_to_numpy(self, dictionary):
        for key in dictionary:
            dictionary[key].detach().numpy()
        return dictionary

    def __call__(self,
                 mutation_file,
                 numpy=True,
                 nworkers=1):
        """Get weights of each signature in lexicographic wrt 1-mer

        Args:
            mutation_vec (np.array(batch_size, 96)): Batch of mutation vectors to decompose
            numpy (bool): Whether to convert the outputs into numpy arrays (otherwise it'd be torch.Tensor). Default: True
            nworkers (int): Num of threads to run the

        Returns:
            results (dict)
        """
        with torch.no_grad():
            # Sort input data columns
            df = pd.read_csv(mutation_file, header=0, index_col=0)
            mutation_order = pd.read_excel(os.path.join(DATA, "mutation_type_order.xlsx"))
            df = df[list(mutation_order['Type'])]
            sample_names = df.index

            mutation_vec = torch.tensor(df.values, dtype=torch.float)
            print(mutation_vec)

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
            results = self.finetuner_errorfinder(mutation_dist=normalized_mutation_vec,
                                                 baseline_guess=self.baseline_guess,
                                                 num_mut=num_mutations.reshape(-1, 1))
            results["normalized_mutation_vec"] = normalized_mutation_vec

        if numpy:
            return self._convert_to_numpy(results), sample_names
        return results, sample_names # I would like the sample_names to be used inside this function to write the files, but needs to be decided