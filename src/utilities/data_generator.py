import os
import sys

import numpy as np
import random
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import csv_to_tensor, read_model
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

    def set_all_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def make_similar_set(self,
                         examples_weight,
                         large_low="low",
                         n_augmentations=10):
        """Create a labelled dataset of mutation vectors similar
           to the provided examples.
        """

        labels = examples_weight
        labels = labels[np.random.permutation(labels.shape[0]), ...]
        labels_sets = {'train':labels[:int(labels.shape[0]*0.7), :],
                       'val':labels[int(labels.shape[0]*0.7):int(labels.shape[0]*0.7+labels.shape[0]*0.05), :],
                       'test':labels[int((labels.shape[0]*0.7+labels.shape[0]*0.05)):, :]}
        labels_output_sets = {'train':0,
                              'val':0,
                              'test':0}
        inputs_sets = {'train':0,
                       'val':0,
                       'test':0}


        for set,labels in labels_sets.items():

            # Perform weight augmentation
            weight_augmenter = WeightAugmenter()
            augmented_labels = weight_augmenter.get_mixed_augmentations(
                weight=labels,
                reweighted_n_augs=int(n_augmentations/2),
                reweighted_augmentation_var=0.3,
                random_n_augs=int(n_augmentations/2),
                random_affected=10,
                random_max_noise=0.3,
            )

            if set == "train":
                if large_low == 'low':
                    range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
                elif large_low == 'large':
                    range_muts = [1e4, 5e4, 1e5, 5e5]
            elif set == "val":
                if large_low == 'low':
                    range_muts = [15, 50, 100, 250, 500, 1000]
                elif large_low == 'large':
                    range_muts = [1e4, 5e4, 1e5, 5e5]
            elif set == "test":
                range_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4, 1e5]

            batch_size = (len(range_muts)-1)*augmented_labels.shape[0]
            input_batch = torch.empty((batch_size, 96))
            labels_batch = torch.empty((batch_size, self.total_signatures + 1))

            for i in tqdm(range(len(range_muts)-1)):
                for j in range(augmented_labels.shape[0]):
                    # Compute resulting signature
                    signature = torch.einsum("ij,j->i", (self.signatures, augmented_labels[j]))

                    if set == "test":
                        num_mut = range_muts[i]
                    else:
                        num_mut = np.random.randint(range_muts[i], range_muts[i+1])

                    if num_mut<1e5:
                        sample = self.__sample_from_sig(signature=signature,
                                                    num_mut=int(num_mut),
                                                    normalize=True)
                        # Store
                        input_batch[i*augmented_labels.shape[0]+j, :] = sample
                        labels_batch[i*augmented_labels.shape[0]+j, :] = torch.cat(
                            [augmented_labels[j, :], torch.tensor([float(num_mut)])])
                    else:
                        # Store
                        input_batch[i*augmented_labels.shape[0]+j, :] = signature
                        labels_batch[i*augmented_labels.shape[0]+j, :] = torch.cat(
                            [augmented_labels[j, :], torch.tensor([float(num_mut)])])
            inputs_sets[set] = input_batch
            labels_output_sets[set] = labels_batch
        return inputs_sets['train'], labels_output_sets['train'],inputs_sets['val'], labels_output_sets['val'],inputs_sets['test'], labels_output_sets['test']

    def make_random_set(self,
                        set,
                        large_low,
                        num_samples,
                        min_n_signatures=1,
                        max_n_signatures=10,
                        normalize=True):
        """Create a labelled dataset of mutation vectors
        randomly combining the signatures.
        """
        assert(min_n_signatures > 0 and max_n_signatures <= self.total_signatures)

        if set == "train":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
                ind_range_muts = [0]*num_samples + [1]*num_samples + [2]*num_samples + \
                    [3] * num_samples + [4]*num_samples + [5]*num_samples + [6]*num_samples + [7]*num_samples
            elif large_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
                # The -1 means real distribution
                ind_range_muts = [0]*num_samples + [1]*num_samples + \
                    [2]*num_samples + [3] * num_samples + [-1]*num_samples
            batch_size = len(ind_range_muts)
        elif set == "val":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
                ind_range_muts = [0]*num_samples + [1]*num_samples + [2]*num_samples + \
                    [3] * num_samples + [4]*num_samples + [5]*num_samples + [6]*num_samples + [7]*num_samples
            elif large_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
                ind_range_muts = [0]*num_samples + [1]*num_samples + [2]*num_samples + \
                    [3]*num_samples + [-1]*num_samples  # The -1 means real distribution
            batch_size = len(ind_range_muts)
        elif set == "test":
            num_muts = [25]*num_samples + [50]*num_samples + [100]*num_samples + [250]*num_samples + [500]*num_samples + [1e3]*num_samples +\
                [5e3]*num_samples + [1e4]*num_samples + [5e4]*num_samples + [1e5]*num_samples
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

    def make_real_set(self, large_low, repetitions=2, normalize=True):
        """Create a labelled dataset of mutation vectors
        from the real weights.
        """

        real_data = csv_to_tensor("../data/real_data/sigprofiler_normalized_PCAWG.csv",
                              device="cpu", header=0, index_col=0)
        real_data = real_data/torch.sum(real_data, axis=1).reshape(-1, 1)
        labels = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)

        labels = labels[np.random.permutation(labels.shape[0]), ...]
        labels_sets = {'train':labels[:int(labels.shape[0]*0.7), :],
                       'val':labels[int(labels.shape[0]*0.7):int(labels.shape[0]*0.7+labels.shape[0]*0.05), :],
                       'test':labels[int((labels.shape[0]*0.7+labels.shape[0]*0.05)):, :]}
        labels_output_sets = {'train':0,
                              'val':0,
                              'test':0}
        inputs_sets = {'train':0,
                       'val':0,
                       'test':0}
        for set,labels in labels_sets.items():
            if set == "train":
                if large_low == 'low':
                    range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
                elif large_low == 'large':
                    range_muts = [1e4, 5e4, 1e5, 5e5]
            elif set == "val":
                if large_low == 'low':
                    range_muts = [15, 50, 100, 250, 500, 1000]
                elif large_low == 'large':
                    range_muts = [1e4, 5e4, 1e5, 5e5]
            elif set == "test":
                range_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4, 1e5]

            for i in range(repetitions-1):
                labels = torch.cat([labels,labels], dim=0)

            batch_size = (len(range_muts)-1)*labels.shape[0]
            input_batch = torch.empty((batch_size, 96))
            labels_batch = torch.empty((batch_size, self.total_signatures + 1))

            for i in tqdm(range(len(range_muts)-1)):
                for j in range(labels.shape[0]):
                    # Compute resulting signature
                    signature = torch.einsum("ij,j->i", (self.signatures, labels[j]))

                    if set == "test":
                        num_mut = range_muts[i]
                    else:
                        num_mut = np.random.randint(range_muts[i], range_muts[i+1])

                    if num_mut<1e5:
                        sample = self.__sample_from_sig(signature=signature,
                                                    num_mut=int(num_mut),
                                                    normalize=normalize)
                        # Store
                        input_batch[i*labels.shape[0]+j, :] = sample
                        labels_batch[i*labels.shape[0]+j, :] = torch.cat(
                            [labels[j, :], torch.tensor([float(num_mut)])])
                    else:
                        # Store
                        input_batch[i*labels.shape[0]+j, :] = signature
                        labels_batch[i*labels.shape[0]+j, :] = torch.cat(
                            [labels[j, :], torch.tensor([float(num_mut)])])
            inputs_sets[set] = input_batch
            labels_output_sets[set] = labels_batch
        return inputs_sets['train'], labels_output_sets['train'],inputs_sets['val'], labels_output_sets['val'],inputs_sets['test'], labels_output_sets['test']

    def augment_set(self, labels):
        labels_changed = labels.detach().clone()
        labels_changed[labels_changed<0.01] = 0
        labels_changed[labels_changed>=0.01] = 1
        prop_labels = (torch.sum(labels_changed, dim=0)/labels_changed.shape[0])

        labels = torch.cat([labels[torch.sum(labels[:, prop_labels < 0.01], dim=1)>0].repeat(5,1),labels], dim=0)
        labels = labels[np.random.permutation(labels.shape[0]), ...]
        return labels

    def augment_real_set(self):
        """Augment the real samples by repeating those
        samples at a small frequency.
        """
        real_data = csv_to_tensor("../data/real_data/sigprofiler_normalized_PCAWG.csv",
                              device="cpu", header=0, index_col=0)
        real_data = real_data/torch.sum(real_data, axis=1).reshape(-1, 1)
        labels = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)

        labels = self.augment_set(labels)
        return labels

    def make_real_set_augmented(self, large_low, repetitions=2, normalize=True):
        """Create a labelled dataset of mutation vectors
        from the real weights augmenting the samples that 
        contains samples at a small frequency.
        """

        real_data = csv_to_tensor("../data/real_data/sigprofiler_normalized_PCAWG.csv",
                              device="cpu", header=0, index_col=0)
        real_data = real_data/torch.sum(real_data, axis=1).reshape(-1, 1)
        labels = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)

        labels_sets = {'train':self.augment_set(labels[:int(labels.shape[0]*0.7), :]),
                       'val':self.augment_set(labels[int(labels.shape[0]*0.7):int(labels.shape[0]*0.7+labels.shape[0]*0.05), :]),
                       'test':self.augment_set(labels[int((labels.shape[0]*0.7+labels.shape[0]*0.05)):, :])}

        labels_output_sets = {'train':0,
                              'val':0,
                              'test':0}
        inputs_sets = {'train':0,
                       'val':0,
                       'test':0}

        for set,labels in labels_sets.items():
            if set == "train":
                if large_low == 'low':
                    range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
                elif large_low == 'large':
                    range_muts = [1e4, 5e4, 1e5, 5e5]
            elif set == "val":
                if large_low == 'low':
                    range_muts = [15, 50, 100, 250, 500, 1000]
                elif large_low == 'large':
                    range_muts = [1e4, 5e4, 1e5, 5e5]
            elif set == "test":
                range_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4, 1e5]

            for i in range(repetitions-1):
                labels = torch.cat([labels,labels], dim=0)

            batch_size = (len(range_muts)-1)*labels.shape[0]
            input_batch = torch.empty((batch_size, 96))
            labels_batch = torch.empty((batch_size, self.total_signatures + 1))

            for i in tqdm(range(len(range_muts)-1)):
                for j in range(labels.shape[0]):
                    # Compute resulting signature
                    signature = torch.einsum("ij,j->i", (self.signatures, labels[j]))

                    if set == "test":
                        num_mut = range_muts[i]
                    else:
                        num_mut = np.random.randint(range_muts[i], range_muts[i+1])

                    if num_mut<1e5:
                        sample = self.__sample_from_sig(signature=signature,
                                                    num_mut=int(num_mut),
                                                    normalize=normalize)
                        # Store
                        input_batch[i*labels.shape[0]+j, :] = sample
                        labels_batch[i*labels.shape[0]+j, :] = torch.cat(
                            [labels[j, :], torch.tensor([float(num_mut)])])
                    else:
                        # Store
                        input_batch[i*labels.shape[0]+j, :] = signature
                        labels_batch[i*labels.shape[0]+j, :] = torch.cat(
                            [labels[j, :], torch.tensor([float(num_mut)])])
            inputs_sets[set] = input_batch
            labels_output_sets[set] = labels_batch
        return inputs_sets['train'], labels_output_sets['train'],inputs_sets['val'], labels_output_sets['val'],inputs_sets['test'], labels_output_sets['test']

    def make_input(self, labels, set, large_low, normalize=True, seed = 0):
        """Create a labelled dataset of mutation vectors
        from a tensor of labels.
        """
        self.set_all_seeds(seed)
        if set == "train":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
            elif large_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
        elif set == "val":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
            elif large_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
        elif set == "test":
            range_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 1]

        batch_size = (len(range_muts)-1)*labels.shape[0]
        input_batch = torch.empty((batch_size, 96))
        labels_batch = torch.empty((batch_size, self.total_signatures + 1))

        for i in tqdm(range(len(range_muts)-1)):
            for j in range(labels.shape[0]):
                # Compute resulting signature
                signature = torch.einsum("ij,j->i", (self.signatures, labels[j]))

                if set == "test":
                    num_mut = range_muts[i]
                else:
                    num_mut = np.random.randint(range_muts[i], range_muts[i+1])

                if num_mut<1e5:
                    sample = self.__sample_from_sig(signature=signature,
                                                num_mut=int(num_mut),
                                                normalize=normalize)
                    # Store
                    input_batch[i*labels.shape[0]+j, :] = sample
                    labels_batch[i*labels.shape[0]+j, :] = torch.cat(
                        [labels[j, :], torch.tensor([float(num_mut)])])
                else:
                    # Store
                    input_batch[i*labels.shape[0]+j, :] = signature
                    labels_batch[i*labels.shape[0]+j, :] = torch.cat(
                        [labels[j, :], torch.tensor([float(num_mut)])])
        return input_batch, labels_batch