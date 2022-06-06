import os
import sys

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

    def __call__(self,
                 mutation_dist,
                 baseline_guess,
                 num_mut):

        classification = self.classifier(mutation_dist=mutation_dist,
                                         num_mut=num_mut).view(-1)
        
        finetuner_guess = self.finetuner(mutation_dist=mutation_dist,
                                         baseline_guess = baseline_guess,
                                         num_mut=num_mut)

        upper, lower = self.errorfinder(weights=finetuner_guess,
                                                      num_mutations=num_mut,
                                                      classification=classification.reshape(-1, 1))
        return finetuner_guess, upper, lower, classification
