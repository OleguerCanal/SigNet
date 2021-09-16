import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_signatures


class WeightAugmenter:
    def __init__(self, signatures):
        self.signatures = torch.stack(signatures).t()
        self._EPS = 1e-6
    
    def __get_distance(self, mut, weight):
        """Return MSE: d(weight) = ||mut - signatures*weight||^2
        """
        return torch.norm(mut - torch.einsum("ij,j->i", self.signatures, weight), p=2)

    def __dist_gradient(self, mut, weight):
        """If using MSE we have: d(weight) = ||mut - signatures*weight||^2
           Thus: d'(weight) = -2*(mut - signatures*weight)^T * signatures
        """
        first_part = -2*(mut - torch.einsum("ij,j->i", self.signatures, weight))
        return torch.einsum("i,ij->j", first_part, self.signatures)

    def get_augmentations(self, mutation_vect, inferred_weight, num_augmentations=10, step_range=(0.0, 0.5)):
        grad = self.__dist_gradient(mut=mutation_vect, weight=inferred_weight)
        print("grad:", grad)
        space = np.linspace(step_range[0], step_range[1], num=10)
        augmentations = [inferred_weight - step*grad for step in space]
        for augmentation in augmentations:
            print(augmentation, self.__get_distance(mutation_vect, augmentation))

        
if __name__== "__main__":
    # signatures = read_signatures("../../data/data.xlsx")
    signatures = [
        torch.tensor([1, 0], dtype=torch.float),
        torch.tensor([0, 1], dtype=torch.float),
        torch.tensor([1, 1], dtype=torch.float),
    ]

    mutation_vect = torch.tensor([2.1, 1.0], dtype=torch.float)
    inferred_weight = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float)

    """
    In this case, we ideally want to obtain things around the line:
    [2.1, 1, 0] + lambda*[1, 1, -1]
    """
    
    weight_augmenter = WeightAugmenter(signatures)
    augmentations = weight_augmenter.get_augmentations(
        mutation_vect=mutation_vect,
        inferred_weight=inferred_weight,
        num_augmentations=5,
        step_range=(0.0, 0.2)
    )