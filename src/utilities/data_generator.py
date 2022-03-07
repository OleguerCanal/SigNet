import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

from models.baseline import Baseline
from utilities.io import csv_to_tensor, read_model
from utilities.weight_augmenter import WeightAugmenter


class DataGenerator:

    def __init__(self,
                 signatures,
                 real_labels=None,
                 seed=None,
                 shuffle=True):
        self.signatures = signatures
        self.total_signatures = signatures.shape[1]
        print("Total signatures:", self.total_signatures)
        if real_labels is not None:
            labels = csv_to_tensor(real_labels, header=0, index_col=0)
            labels = labels/torch.sum(labels, axis=1).reshape(-1, 1)
            self.real_labels = torch.cat([labels, torch.zeros(labels.size(0), 7).to(labels)], dim=1)
        else:
            self.real_labels = None

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
                         examples_weight,
                         large_or_low="low",
                         is_test=False,
                         n_augmentations=10):
        """Create a labelled dataset of mutation vectors similar
           to the provided examples.
        """

        # Perform weight augmentation
        weight_augmenter = WeightAugmenter()
        augmented_labels = weight_augmenter.get_mixed_augmentations(
            weight=examples_weight,
            reweighted_n_augs=int(n_augmentations/2),
            reweighted_augmentation_var=0.3,
            random_n_augs=int(n_augmentations/2),
            random_affected=10,
            random_max_noise=0.3,
        )

        # Sampling:
        batch_size = augmented_labels.shape[0]
        mutations = torch.einsum(
            "ij,bj->bi", (self.signatures, augmented_labels))

        if not is_test:
            if large_or_low == 'low':
                partitions_points = int(batch_size/7)+1
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 10000]
                ind_range_muts = [0]*partitions_points + [1]*partitions_points + [2]*partitions_points + [
                    3] * partitions_points + [4]*partitions_points + [5]*partitions_points + [6]*(partitions_points+1)
            elif large_or_low == 'large':
                partitions_points = int(batch_size/5)
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
                ind_range_muts = [0]*partitions_points + [1]*partitions_points + [2]*partitions_points + [
                    3] * partitions_points + [-1]*(partitions_points+1)     # The -1 means real distribution
        else:
            num_muts = [25]*100 + [50]*100 + [100]*100 + [250]*100 + [500]*100 + [1e3]*100 +\
                [5e3]*100 + [1e4]*100 + [5e4]*100 + [1e5] * \
                100    # The -1 means real distribution
            batch_size = len(num_muts)

        input_batch = torch.empty((batch_size, 96))
        label_batch = torch.empty((batch_size, self.total_signatures + 1))

        for i in tqdm(range(batch_size)):
            label = augmented_labels[i, ...]
            signature = mutations[i, ...]

            # Sample (NOTE: This bit is a mess, we should rethink it)
            if is_test:
                num_mut = num_muts[i]
            else:
                if ind_range_muts[i] != -1:
                    num_mut = np.random.randint(
                        range_muts[ind_range_muts[i]], range_muts[ind_range_muts[i] + 1])
                else:
                    num_mut = -1

            if num_mut != -1:
                sample = self.__sample_from_sig(signature=signature,
                                                num_mut=int(num_mut),
                                                normalize=True)
                # Store
                input_batch[i, :] = sample
                label_batch[i, :] = torch.cat(
                    [label, torch.tensor([float(num_mut)])])
            else:
                # Store
                input_batch[i, :] = signature
                # For the real distribution we say we have more than 1e5 mutations
                label_batch[i, :] = torch.cat(
                    [label, torch.tensor([float(np.random.randint(1e5, 1e6))])])

        if self.shuffle:
            indices = np.random.permutation(input_batch.shape[0])
            input_batch = input_batch[indices, ...]
            label_batch = label_batch[indices, ...]

        return input_batch, label_batch

    def make_random_set(self,
                        set,
                        large_low,
                        min_n_signatures=1,
                        max_n_signatures=10,
                        normalize=True):
        """Create a labelled dataset of mutation vectors
        randomly combining the signatures.
        """
        assert(min_n_signatures > 0 and max_n_signatures <= self.total_signatures)

        if set == "train":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 10000]
                ind_range_muts = [0]*50000 + [1]*50000 + [2]*50000 + \
                    [3] * 50000 + [4]*50000 + [5]*50000 + [6]*50000
            elif large_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
                # The -1 means real distribution
                ind_range_muts = [0]*50000 + [1]*50000 + \
                    [2]*50000 + [3] * 50000 + [-1]*50000
            batch_size = len(ind_range_muts)
        elif set == "val":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000]
            elif large_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
            ind_range_muts = [0]*1000 + [1]*1000 + [2]*1000 + \
                [3]*1000 + [-1]*1000  # The -1 means real distribution
            batch_size = len(ind_range_muts)
        elif set == "test":
            num_muts = [25]*100 + [50]*100 + [100]*100 + [250]*100 + [500]*100 + [1e3]*100 +\
                [5e3]*100 + [1e4]*100 + [5e4]*100 + [1e5] * \
                100    # The -1 means real distribution
            batch_size = len(num_muts)

        input_batch = torch.empty((batch_size, 96))
        label_batch = torch.empty((batch_size, self.total_signatures + 1))

        for i in tqdm(range(batch_size)):
            # Pick the number of involved signatures
            n_signatures = np.random.randint(
                min_n_signatures, max_n_signatures + 1)

            # Select n_signatures
            signature_ids = torch.randperm(self.total_signatures)[
                :n_signatures]

            # Assign weights randomly
            weights = torch.rand(size=(n_signatures,))
            weights = weights/torch.sum(weights)

            # We want weights larger than 0.1
            for j in range(len(weights)):
                if weights[j] < 0.1:
                    weights[j] = 0
            weights = weights/torch.sum(weights)
            label = torch.zeros(self.total_signatures).scatter_(
                dim=0, index=signature_ids, src=weights)

            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, label))

            # Sample (NOTE: This bit is a mess, we should rethink it)
            if set == "test":
                num_mut = num_muts[i]
            else:
                if ind_range_muts[i] != -1:
                    num_mut = np.random.randint(
                        range_muts[ind_range_muts[i]], range_muts[ind_range_muts[i] + 1])
                else:
                    num_mut = -1

            if num_mut != -1:
                sample = self.__sample_from_sig(signature=signature,
                                                num_mut=int(num_mut),
                                                normalize=normalize)
                # Store
                input_batch[i, :] = sample
                label_batch[i, :] = torch.cat(
                    [label, torch.tensor([float(num_mut)])])
            else:
                # Store
                input_batch[i, :] = signature
                # For the real distribution we say we have more than 1e5 mutations
                label_batch[i, :] = torch.cat(
                    [label, torch.tensor([float(np.random.randint(1e5, 1e6))])])
        return input_batch, label_batch

    def make_realistic_set(self,
                           generator_model_path,
                           set,
                           large_low,
                           std = 1.0, 
                           normalize=True):
        """Create a labelled dataset of mutation vectors
        from the generator output weights.
        """
        generator = read_model(generator_model_path, device="cpu")

        if set == "train":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
                ind_range_muts = [0]*50000 + [1]*50000 + [2]*50000 +\
                    [3]*50000 + [4]*50000 + [5]*50000 + [6]*50000 +\
                    [7]*50000 + [-1]*50000
            elif large_low == 'large':
                range_muts = [1e4, 5e4, 1e5, 5e5]
                # The -1 means real distribution
                ind_range_muts = [0]*100000 + [1]*100000 + [-1]*100000
            batch_size = len(ind_range_muts)
        elif set == "val":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000]
            elif large_low == 'large':
                range_muts = [1e4, 5e4, 1e5, 5e5]
            ind_range_muts = [0]*1000 + [1]*1000 + [-1]*1000  # The -1 means real distribution
            batch_size = len(ind_range_muts)
        elif set == "test":
            num_muts = [25]*1000 + [50]*1000 + [100]*1000 + [250]*1000 + [500]*1000 + [1e3]*1000 +\
                [5e3]*1000 + [1e4]*1000 + [5e4]*1000 + [1e5] * \
                1000    # The -1 means real distribution
            batch_size = len(num_muts)

        labels = generator.generate(batch_size, std = std)
        if self.real_labels is not None:
            labels = generator.filter(labels, self.real_labels)
            labels = torch.cat([labels, self.real_labels], dim=0)
        
        input_batch = torch.empty((batch_size, 96))
        labels_batch = torch.empty((batch_size, self.total_signatures + 1))

        for i in tqdm(range(batch_size)):
            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, labels[i]))

            # Sample (NOTE: This bit is a mess, we should rethink it)
            if set == "test":
                num_mut = num_muts[i]
            else:
                if ind_range_muts[i] != -1:
                    num_mut = np.random.randint(
                        range_muts[ind_range_muts[i]], range_muts[ind_range_muts[i] + 1])
                else:
                    num_mut = -1

            if num_mut != -1:
                sample = self.__sample_from_sig(signature=signature,
                                                num_mut=int(num_mut),
                                                normalize=normalize)
                # Store
                input_batch[i, :] = sample
                labels_batch[i, :] = torch.cat(
                    [labels[i, :], torch.tensor([float(num_mut)])])
            else:
                # Store
                input_batch[i, :] = signature
                # For the real distribution we say we have more than 1e5 mutations
                labels_batch[i, :] = torch.cat(
                    [labels[i, :], torch.tensor([float(np.random.randint(1e5, 1e6))])])
        return input_batch, labels_batch

    def make_real_set(self):
        """Create a labelled dataset of mutation vectors
        from the real weights.
        """

        real_data = csv_to_tensor("../data/real_data/sigprofiler_normalized_PCAWG.csv",
                              device="cpu", header=0, index_col=0)
        real_data = real_data/torch.sum(real_data, axis=1).reshape(-1, 1)
        labels = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)

        batch_size = labels.shape[0]
        input_batch = torch.empty((batch_size, 96))
        labels_batch = torch.empty((batch_size, self.total_signatures + 1))

        for i in tqdm(range(batch_size)):
            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, labels[i]))

            num_mut = float(np.random.randint(1e3, 1e4))
            sample = self.__sample_from_sig(signature=signature,
                                            num_mut=int(num_mut),
                                            normalize=True)
            # Store
            input_batch[i, :] = sample
            labels_batch[i, :] = torch.cat(
                [labels[i, :], torch.tensor([float(num_mut)])])

        return input_batch, labels_batch