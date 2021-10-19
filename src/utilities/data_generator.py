import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

from models.baseline import Baseline
from utilities.weight_augmenter import WeightAugmenter

class DataGenerator:

    def __init__(self,
                 signatures,
                 seed=None,
                 shuffle=True):
        self.signatures = signatures
        self.total_signatures = signatures.shape[1]
        print("Total signatures:", self.total_signatures)
        self.shuffle = shuffle
        if seed is not None:
            torch.seed = seed
            np.random.seed(seed)

    def __sample_from_sig(self, signature, num_mut, normalize=True):
        sample = signature
        if num_mut < 1e5:  # only do sampling for num_mut < 1e5
            c = torch.distributions.categorical.Categorical(probs=signature)
            sample_shape = torch.Size([num_mut, ])
            samples = c.sample(sample_shape=sample_shape).type(torch.float32)
            n_bins = signature.shape[0]
            sample = torch.histc(samples, bins=n_bins, min=0, max=n_bins - 1)
        if normalize:
            sample = sample/float(num_mut)
        return sample

    def make_similar_set(self,
                         examples_input,
                         n_augmentations=10):
        """Create a labelled dataset of mutation vectors similar
           to the provided examples.
        """
        # Get initial guess
        baseline = Baseline(self.signatures)
        baseline_guess = baseline.get_weights_batch(examples_input)

        # Perform weight augmentation
        weight_augmenter = WeightAugmenter()
        augmented_labels = weight_augmenter.get_mixed_augmentations(
            weight=baseline_guess,
            reweighted_n_augs=int(n_augmentations/2),
            reweighted_augmentation_var=0.3,
            random_n_augs=int(n_augmentations/2),
            random_prop_affected=6./72.,
            random_max_noise=0.2,
        )

        # Sampling:
        num_datapoints = augmented_labels.shape[0]
        mutations = torch.einsum("ij,bj->bi", (self.signatures, augmented_labels))
        range_muts = [15, 50, 75, 100, 150, 250, 500, 1e3, 1e4, 1e5, 1e6]
        data_input = torch.empty((num_datapoints*(len(range_muts)-1), examples_input.shape[1]))
        data_label = torch.empty((num_datapoints*(len(range_muts)-1), augmented_labels.shape[1] + 1))
        for i in range(len(range_muts)-1):
            for j in range(num_datapoints):
                num_mut = np.random.randint(range_muts[i], range_muts[i+1])
                data_input[j+i*num_datapoints,:] = self.__sample_from_sig(signature=mutations[j, :],
                                                                          num_mut=num_mut)
                data_label[j+i*num_datapoints,:] = torch.cat([augmented_labels[j, :], torch.tensor([num_mut])])

        if self.shuffle:
            indices = np.random.permutation(data_input.shape[0])
            data_input = data_input[indices, ...]
            data_label = data_label[indices, ...]
        
        return data_input, data_label

    def make_random_set(self,
                        set,
                        min_n_signatures=1,
                        max_n_signatures=10,
                        normalize=True):
        """Create a labelled dataset of mutation vectors
        randomly combining the signatures.
        """
        assert(min_n_signatures > 0 and max_n_signatures <= self.total_signatures)

        if set == "train":
            range_muts = [15, 100, 500, 5000, 50000]
            ind_range_muts = [0]*30000 + [1]*30000 + [2]*20000 + [3] * \
                20000 + [-1]*20000     # The -1 means real distribution
            batch_size = len(ind_range_muts)
        elif set == "val":
            range_muts = [15, 100, 500, 5000, 50000]
            # The -1 means real distribution
            ind_range_muts = [0]*3000 + [1]*3000 + \
                [2]*2000 + [3]*2000 + [-1]*2000
            batch_size = len(ind_range_muts)
        elif set == "test":
            num_muts = [25]*1000 + [50]*1000 + [100]*1000 + [150]*1000 +\
                [200]*1000 + [250]*1000 + [500]*1000 + [1000]*1000 +\
                [2000]*1000 + [5000]*1000 + [10000]*1000 + [20000] * 1000 +\
                [50000]*1000 + [-1]*2000    # The -1 means real distribution
            batch_size = len(num_muts)
        
        input_batch = torch.empty((batch_size, 96))
        label_batch = torch.empty((batch_size, self.total_signatures + 1))

        for i in range(batch_size):
            # Pick the number of involved signatures
            n_signatures = np.random.randint(min_n_signatures, max_n_signatures + 1)

            # Select n_signatures
            signature_ids = torch.randperm(self.total_signatures)[:n_signatures]

            # Assign weights randomly
            weights = torch.rand(size=(n_signatures,))
            weights = weights/torch.sum(weights)

            # We want weights larger than 0.1
            for j in range(len(weights)):
                if weights[j] < 0.1:
                    weights[j] = 0
            weights = weights/torch.sum(weights)
            label = torch.zeros(self.total_signatures).scatter_(dim=0, index=signature_ids, src=weights)

            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, label))

            # Sample
            if ind_range_muts[i] != -1:
                num_mut = np.random.randint(range_muts[ind_range_muts[i]], range_muts[ind_range_muts[i] + 1])
            else:
                num_mut = -1

            if set == "test":
                num_mut = num_muts[i]

            if num_mut != -1:
                sample = self.__sample_from_sig(signature=signature,
                                                num_mut=num_mut,
                                                normalize=normalize)
                # Store
                input_batch[i, :] = sample
                label_batch[i, :] = torch.cat([label, torch.tensor([num_mut])])
            else:
                # Store
                input_batch[i, :] = signature
                # For the real distribution we say we have more than 1e5 mutations
                label_batch[i, :] = torch.cat(
                    [label, torch.tensor([np.random.randint(1e5, 1e6)])])
            if i % 1000 == 0:
                print(i)

        return input_batch, label_batch
