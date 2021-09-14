import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.weights_to_mutation import WeightToMutation

class Generator(nn.Module):

    def __init__(self, baseline, finetuner, signatures):
        super(Generator, self).__init__()
        self.__baseline = baseline
        self.__finetuner = finetuner
        self.__w2m = WeightToMutation(signatures=signatures)

    def forward(self, mutation_dist, num_mut):
        finetuned_weights = self.get_finetuned_weights(mutation_dist, num_mut)
        guessed_mutation = self.__w2m.get_mutation(finetuned_weights, noise_variance=0.0)
        return guessed_mutation

    def get_finetuned_weights(self, mutation_dist, num_mut):
        baseline_weights = self.__baseline.get_weights_batch(mutation_dist)
        finetuned_weights = self.__finetuner(mutation_dist, baseline_weights, num_mut)
        return finetuned_weights