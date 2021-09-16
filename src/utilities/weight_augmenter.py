import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_signatures


class WeightAugmenter:
    def __init__(self):
        self._cutoff = 0.05

    def __normalize(self, weights):
        return weights/torch.sum(weights, axis=1).view(-1, 1)

    def get_reweighted_augmentations(self, weight, num_augmentations=5, augmentation_variance=0.5):
        """Applies normal weight changes with variance augmentation_variance to the batch of weights
        with weight higher than a minimum threshold.

        Args:
            weight (torch.tensor(batch,num_sigs])): Batch of guessed weights
            num_augmentations (int, optional): Number of variations per weight in batch
        
        Returns:
            torch.tensor: Tensor of augmentations
        """
        mask = (weight > self._cutoff).type(torch.int).float()
        augmented_weights = weight.repeat(num_augmentations, 1)
        augmented_mask = mask.repeat(num_augmentations, 1)
        augmented_noise = augmentation_variance*augmented_mask*torch.rand_like(augmented_mask)
        augmented_weights = torch.max(torch.zeros_like(augmented_weights), augmented_weights + augmented_noise)
        return self.__normalize(augmented_weights)

    def get_random_augmentations(self, weight, num_augmentations=5, prop_weights_affected=6./72., max_noise=0.3):
        """Apply random weight upscaling to a proportion of prop_weights_affected weights with a uniform
        value from 0 to max_noise.

        Args:
            weight (torch.tensor(batch,num_sigs)): Batch of guessed weights
            num_augmentations (int, optional): Number of variations per weight in batch
            prop_weights_affected (float, optional): Provability of changing a weight. Defaults to 6./72..
            max_noise (float, optional): Maximum weight upscaling. Defaults to 0.3.

        Returns:
            torch.tensor: Tensor of augmentations
        """
        augmented_weights = weight.repeat(num_augmentations, 1)
        mask = (torch.rand(augmented_weights.size()) > 1 - prop_weights_affected)
        noise = max_noise*torch.rand(augmented_weights.size())*mask
        augmented_weights = augmented_weights + noise
        return self.__normalize(augmented_weights)

if __name__== "__main__":
    weight_batch = torch.tensor([[0.1, 0.9, 0], [0.3, 0.3, 0.4]])
    
    weight_augmenter = WeightAugmenter()
    augmentations = weight_augmenter.get_reweighted_augmentations(
        weight=weight_batch,
        num_augmentations=2,
    )
    print(augmentations)
    
    augmentations = weight_augmenter.get_random_augmentations(
        weight=weight_batch,
        num_augmentations=2,
        prop_weights_affected=1./4.
    )
    print(augmentations)
