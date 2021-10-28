import os
import sys

import numpy as np
import torch

from utilities.io import read_model

class ClassifiedFinetunerErrorfinder:

    def __init__(self,
                 classifier,
                 realistic_finetuner,
                 random_finetuner,
                 realistic_errorfinder,
                 random_errorfinder,
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

        self.realistic_errorfinder = read_model(realistic_errorfinder)
        self.random_errorfinder = read_model(random_errorfinder)

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
        real_error_upper, real_error_lower = self.realistic_errorfinder(real_guess, num_mut_real)

        # Select and finetune mutations classified as random
        mut_dist_rand = mutation_dist[ind_rand, ...]
        weights_rand = weights[ind_rand, ...]
        num_mut_rand = num_mut[ind_rand]
        rand_guess = self.random_finetuner(mutation_dist=mut_dist_rand,
                                           baseline_guess=weights_rand,
                                           num_mut=num_mut_rand)
        rand_error_upper, rand_error_lower = self.random_errorfinder(rand_guess, num_mut_rand)

        # Join predictions and re-order them as originally
        joined_guess = torch.cat((real_guess, rand_guess), dim=0)
        joined_upper = torch.cat((real_error_upper, rand_error_upper), dim=0)
        joined_lower = torch.cat((real_error_lower, rand_error_lower), dim=0)
        ind_order = [float(el) for el in ind_order]
        joined_guess = torch.cat(
            (joined_guess, torch.tensor(ind_order).reshape(-1, 1)), dim=1)
        joined_guess = joined_guess[joined_guess[:, -1].sort()[1]]
        joined_guess = joined_guess[:, :-1]

        joined_upper = torch.cat(
            (joined_upper, torch.tensor(ind_order).reshape(-1, 1)), dim=1)
        joined_upper = joined_upper[joined_upper[:, -1].sort()[1]]
        joined_upper = joined_upper[:, :-1]

        joined_lower = torch.cat(
            (joined_lower, torch.tensor(ind_order).reshape(-1, 1)), dim=1)
        joined_lower = joined_lower[joined_lower[:, -1].sort()[1]]
        joined_lower = joined_lower[:, :-1]
        return joined_guess, joined_upper, joined_lower
