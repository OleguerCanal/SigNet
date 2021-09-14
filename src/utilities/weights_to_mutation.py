import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_signatures


class WeightToMutation:
    def __init__(self, signatures):
        self.signatures = torch.stack(signatures).t()
        self._EPS = 1e-6

    def get_mutation(self, weights, noise_variance=0.0):
        mutation = torch.einsum("ji,bi->bj", self.signatures, weights)
        if noise_variance > 0.0:  # NOTE(oleguer): This bloc sometimes has NAN issues not sure why
            mask = (mutation > self._EPS).type(torch.int).float()
            noise = noise_variance*torch.rand_like(mutation)
            mutation = torch.abs(mutation + noise*mask)
            mutation = mutation/torch.sum(mutation, axis=1).view(-1, 1)
        return mutation

        
if __name__== "__main__":
    signatures = read_signatures("../../data/data.xlsx")
    w2m = WeightToMutation(signatures)
    w = torch.rand(2, 72)
    mutation = w2m.get_mutation(w)
    print(mutation.shape)
