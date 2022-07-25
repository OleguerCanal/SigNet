import logging
import numpy as np
import torch 

class ClassifiedFinetunerErrorfinder:

    def __init__(self,
                 classifier,
                 finetuner,
                 errorfinder,
                 classification_cutoff=0.5,
                 device="cpu"):
        """Instantiate a ClassifiedFinetuner

        Args:
            classifier (Classifier): Model to discriminate between realistic and random data
            realistic_finetuner (Finetuner or CombinedFinetuner): Model which improves baseline guess for random data
            random_finetuner (Finetuner or CombinedFinetuner): Model which improves baseline guess for random data
            classification_cutoff (float, optional): Cuttoff at which we decide something is realistic. Defaults to 0.5.
            device (str, optional): Device to use (cuda or cpu). Defaults to "cpu".
        """

        self.classification_cutoff = classification_cutoff
        self.device = device

        self.classifier = classifier
        self.finetuner = finetuner
        self.errorfinder = errorfinder

    def __separate_classification(self, classification, mutation_dist, baseline_guess, num_mut):
        ind = torch.tensor(range(classification.size()[0]))
        ind_order = torch.tensor(np.concatenate((ind[classification <= self.classification_cutoff], ind[classification > self.classification_cutoff]))).reshape(-1, 1).to(torch.float).to(self.device)
        
        input_batch_random = mutation_dist[classification <= self.classification_cutoff, ]
        input_batch_realistic = mutation_dist[classification > self.classification_cutoff, ]
        num_mut_realistic = num_mut[classification > self.classification_cutoff, ]
        classification_realistic = classification[classification > self.classification_cutoff, ]

        baseline_guess_random = baseline_guess[classification <= self.classification_cutoff, ]
        baseline_guess_realistic = baseline_guess[classification > self.classification_cutoff, ]

        return input_batch_realistic, input_batch_random, baseline_guess_random, baseline_guess_realistic, num_mut_realistic, classification_realistic, ind_order

    def __join_and_sort(self, realistic, random, ind_order):
        joined = torch.cat((realistic, random), dim=0)
        joined = torch.cat((joined, ind_order), dim=1)
        joined = joined[joined[:, -1].sort()[1]]
        return joined[:, :-1]

    def _apply_cutoff(self, comb, cutoff):    #Think about the issue when baseline is not normalized
        mask = (comb > cutoff).type(torch.int).float()
        comb = comb*mask
        comb = torch.cat((comb, torch.ones_like(torch.sum(
        comb, axis=1).reshape((-1, 1)))-torch.sum(
        comb, axis=1).reshape((-1, 1))), axis=1)
        return comb

    def __call__(self,
                 mutation_dist,
                 baseline_guess,
                 num_mut):

        logging.info("Detecting out-of-train-distribution points...")
        classification = self.classifier(mutation_dist=mutation_dist,
                                         num_mut=num_mut).view(-1)
        logging.info("Detecting out-of-train-distribution points... DONE")
        
        mutation_dist_realistic, mutation_dist_random, baseline_guess_random, baseline_guess_realistic, num_mut_realistic, classification_realistic, ind_order = self.__separate_classification(classification, mutation_dist, baseline_guess, num_mut)

        logging.info("Finetuning NNLS guesses...")
        finetuner_guess_realistic = self.finetuner(mutation_dist=mutation_dist_realistic,
                                         baseline_guess = baseline_guess_realistic,
                                         num_mut=num_mut_realistic)

        baseline_guess_random = self._apply_cutoff(baseline_guess_random, 0.01)

        finetuner_guess = self.__join_and_sort(finetuner_guess_realistic, baseline_guess_random, ind_order)
        logging.info("Finetuning NNLS guesses... DONE")

        logging.info("Estimating errorbars...")
        upper, lower = self.errorfinder(weights=finetuner_guess_realistic[:,:-1],
                                                num_mutations=num_mut_realistic,
                                                classification=classification_realistic.reshape(-1, 1))
        upper = self.__join_and_sort(upper, torch.full_like(baseline_guess_random[:,:-1], float('nan')), ind_order)
        lower = self.__join_and_sort(lower, torch.full_like(baseline_guess_random[:,:-1], float('nan')), ind_order)
        logging.info("Estimating errorbars... DONE")

        result = {"finetuner_guess": finetuner_guess,
                  "error_upper": upper,
                  "error_lower": lower,
                  "classification": classification}
        return result