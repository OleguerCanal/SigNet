import os
import sys

import torch
import numpy as np


class WeightAugmenter:
    def __init__(self):
        self._cutoff = 0.05

    def __normalize(self, weights):
        return weights/torch.sum(weights, axis=1).view(-1, 1)

    def get_reweighted_augmentations(self,
                                     weight,
                                     num_augmentations=5,
                                     augmentation_variance=0.5):
        """Applies normal weight changes with variance augmentation_variance
        to the batch of weights with weight higher than a minimum threshold.

        Args:
            weight (torch.tensor(batch,num_sigs])): Batch of guessed weights
            num_augmentations (int, optional): Number of variations per weight in batch

        Returns:
            torch.tensor: Tensor of augmentations
        """
        mask = (weight > self._cutoff).type(torch.int).float()
        augmented_weights = weight.repeat(num_augmentations, 1)
        augmented_mask = mask.repeat(num_augmentations, 1)
        augmented_noise = augmentation_variance * augmented_mask*torch.rand_like(augmented_mask)
        augmented_weights = torch.max(torch.zeros_like(augmented_weights), augmented_weights + augmented_noise)
        augmented_weights = augmented_weights[torch.sum(augmented_weights, axis=1) > 0, ...]
        return self.__normalize(augmented_weights)

    def get_random_augmentations(self,
                                 weight,
                                 num_augmentations=5,
                                 prop_weights_affected=6./72.,
                                 max_noise=0.3):
        """Apply random weight upscaling to a proportion of 
        prop_weights_affected weights with a uniform
        value from 0 to max_noise.

        Args:
            weight (torch.tensor(batch,num_sigs)): Batch of guessed weights
            num_augmentations (int, optional): Number of variations per weight in batch
            prop_weights_affected (float, optional): Provability of changing a weight.
            max_noise (float, optional): Maximum weight upscaling.

        Returns:
            torch.tensor: Tensor of augmentations
        """
        augmented_weights = weight.repeat(num_augmentations, 1)
        mask = (torch.rand(augmented_weights.size()) > 1 - prop_weights_affected)
        noise = max_noise*torch.rand(augmented_weights.size())*mask
        augmented_weights = augmented_weights + noise
        return self.__normalize(augmented_weights)

    def get_mixed_augmentations(self,
                                weight,
                                reweighted_n_augs=5,
                                reweighted_augmentation_var=0.5,
                                random_n_augs=5,
                                random_prop_affected=6./72.,
                                random_max_noise=0.3):
        reweighted_augmentations = self.get_reweighted_augmentations(
            weight=weight,
            num_augmentations=reweighted_n_augs,
            augmentation_variance=reweighted_augmentation_var
        )
        random_augmentations = self.get_random_augmentations(
            weight=weight,
            num_augmentations=random_n_augs,
            prop_weights_affected=random_prop_affected,
            max_noise=random_max_noise
        )
        augmentations = torch.cat([reweighted_augmentations, random_augmentations])
        return augmentations[np.random.permutation(augmentations.shape[0]), ...]


if __name__ == "__main__":
    weight_batch = torch.tensor([[0.1, 0.9, 0], [0.3, 0.3, 0.4]])

    weight_augmenter = WeightAugmenter()
    # augmentations = weight_augmenter.get_reweighted_augmentations(
    #     weight=weight_batch,
    #     num_augmentations=2,
    # )
    # print(augmentations)

    # augmentations = weight_augmenter.get_random_augmentations(
    #     weight=weight_batch,
    #     num_augmentations=2,
    #     prop_weights_affected=1./4.
    # )
    # print(augmentations)

    augmentations = weight_augmenter.get_mixed_augmentations(
        weight=weight_batch,
        reweighted_n_augs=2,
        random_n_augs=1,
        random_prop_affected=0.5,
    )
    print(augmentations)
