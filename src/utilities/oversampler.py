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