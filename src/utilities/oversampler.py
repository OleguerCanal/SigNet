import copy

import numpy as np
import torch

class OverSampler:
    def __init__(self, data):
        self.data = data
        self.presence = self._get_presence(data)

    def _get_presence(self, data, offset = 0.05):
        presence = torch.sum(data, dim=0)
        return presence / torch.sum(presence)

    def get_frequencies(self, points):
        frequencies = torch.einsum("i,bi->b", self.presence, points)
        return frequencies

    def get_oversampled_set(self, rarity=0.1, n_repetitions=5):
        """Take the rarity% and repeats it n_repetitions
        """
        oversampled = copy.deepcopy(self.data)
        frequencies = self.get_frequencies(oversampled)
        sorted_frequencies, indices = torch.sort(frequencies)
        to_repeat = oversampled[indices[:int(rarity*indices.size(0))], ...]
        to_append = [to_repeat]*n_repetitions
        oversampled = torch.cat([oversampled] + to_append, dim=0)
        return oversampled[np.random.permutation(oversampled.size(0)), ...]


class CancerTypeOverSampler:
    def __init__(self, data, cancer_types):
        self.data = data
        self.cancer_types = cancer_types
        self.counts = torch.bincount(cancer_types.to(torch.int))
        # print(cancer_types)
        # print(self.counts)

    def get_oversampled_set(self, rarity=0.1, n_repetitions=5):
        """Take the rarity% and repeats it n_repetitions
        """
        oversampled = copy.deepcopy(self.data)
        uniform_count = int(self.cancer_types.size(0)/self.counts.size(0))
        for i in range(self.cancer_types.size(0)):
            count = self.counts[int(self.cancer_types[i].item())]
            times_to_repeat = min(int(torch.round(uniform_count/count)), 10)  # dont repeat more than 10 times
            # print(i, int(self.cancer_types[i].item()), count, times_to_repeat)
            to_append = [self.data[i, :].unsqueeze(0)]*times_to_repeat
            oversampled = torch.cat([oversampled] + to_append, dim=0)
        return oversampled[np.random.permutation(oversampled.size(0)), ...]