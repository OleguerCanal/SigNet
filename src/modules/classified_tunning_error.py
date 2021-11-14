import os
import sys

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

        self.realistic_errorfinder = realistic_errorfinder
        self.random_errorfinder = random_errorfinder

    def __join_and_sort(self, real, rand, ind_order):
        joined = torch.cat((real, rand), dim=0)
        joined = torch.cat((joined, ind_order), dim=1)
        joined = joined[joined[:, -1].sort()[1]]
        return joined[:, :-1]

    def __call__(self,
                 mutation_dist,
                 weights,
                 num_mut):
        # Classify inputs depending on whether they come from a real distribution or not
        classification = self.classifier(mutation_dist=mutation_dist,
                                         num_mut=num_mut).view(-1)
        
        # This is for debugging purposes
        self.classification_results = (classification >= self.classification_cutoff).to(torch.int64)

        # Remember input indexes
        ind = torch.tensor(range(mutation_dist.size()[0]))
        ind_real = ind[classification >= self.classification_cutoff]
        ind_rand = ind[classification < self.classification_cutoff]
        ind_order = torch.tensor(np.concatenate((ind_real, ind_rand))).reshape(-1, 1).to(torch.float)

        # Select and finetune mutations classified as real
        mut_dist_real = mutation_dist[ind_real, ...]
        weights_real = weights[ind_real, ...]
        num_mut_real = num_mut[ind_real]
        # print("real", mut_dist_real.shape)
        real_guess = self.realistic_finetuner(mutation_dist=mut_dist_real,
                                              baseline_guess=weights_real,
                                              num_mut=num_mut_real)
        real_error_upper, real_error_lower = self.realistic_errorfinder(real_guess, num_mut_real)

        # Select and finetune mutations classified as random
        mut_dist_rand = mutation_dist[ind_rand, ...]
        weights_rand = weights[ind_rand, ...]
        num_mut_rand = num_mut[ind_rand]
        # print("rand", mut_dist_rand.shape)
        rand_guess = self.random_finetuner(mutation_dist=mut_dist_rand,
                                           baseline_guess=weights_rand,
                                           num_mut=num_mut_rand)
        rand_error_upper, rand_error_lower = self.random_errorfinder(rand_guess, num_mut_rand)

        # Join predictions and re-order them as originally
        joined_guess = self.__join_and_sort(real=real_guess, rand=rand_guess, ind_order=ind_order)
        joined_upper = self.__join_and_sort(real=real_error_upper, rand=rand_error_upper, ind_order=ind_order)
        joined_lower = self.__join_and_sort(real=real_error_lower, rand=rand_error_lower, ind_order=ind_order)
        return joined_guess, joined_upper, joined_lower
