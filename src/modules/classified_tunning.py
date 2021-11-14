import os
import sys

import numpy as np
import torch

from utilities.io import read_model

class ClassifiedFinetuner:

    def __init__(self,
                 classifier,
                 realistic_finetuner,
                 random_finetuner,
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
        self.realistic_finetuner = realistic_finetuner
        self.random_finetuner = random_finetuner

    def __call__(self,
                 mutation_dist,
                 weights,
                 num_mut):
        # Classify inputs depending on whether they come from a real distribution or not
        classification = self.classifier(mutation_dist=mutation_dist,
                                         num_mut=num_mut).view(-1)

        # Remember input indexes
        ind = np.array(range(mutation_dist.size()[0]))
        ind_real = ind[classification >= self.classification_cutoff]
        ind_rand = ind[classification < self.classification_cutoff]
        ind_order = np.concatenate((ind_real, ind_rand))

        # Select and finetune mutations classified as real
        mut_dist_real = mutation_dist[ind_real, ...]
        weights_real = weights[ind_real, ...]
        num_mut_real = num_mut[ind_real]
        real_guess = self.realistic_finetuner(mutation_dist=mut_dist_real,
                                              baseline_guess=weights_real,
                                              num_mut=num_mut_real)

        # Select and finetune mutations classified as random
        mut_dist_rand = mutation_dist[ind_rand, ...]
        weights_rand = weights[ind_rand, ...]
        num_mut_rand = num_mut[ind_rand]
        rand_guess = self.random_finetuner(mutation_dist=mut_dist_rand,
                                           baseline_guess=weights_rand,
                                           num_mut=num_mut_rand)

        # Join predictions and re-order them as originally
        joined_guess = torch.cat((real_guess, rand_guess), dim=0)
        ind_order = torch.from_numpy(ind_order).type(torch.float).reshape(-1, 1)
        joined_guess = torch.cat((joined_guess, ind_order), dim=1)
        joined_guess = joined_guess[joined_guess[:, -1].sort()[1]]
        joined_guess = joined_guess[:, :-1]
        return joined_guess, classification.reshape(-1,1)
