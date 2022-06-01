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
        self.shuffle = shuffle
        self.set_all_seeds(seed)
        print("Total signatures:", self.total_signatures)

    def set_all_seeds(self, seed):
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def __get_nummuts(self, split, large_or_low):
        assert split in ["train", "valid", "test"]
        assert large_or_low in ["large", "low"]

        # For train and validation we sample from ranges
        if split == "train" or split == "valid":
            if large_or_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
                ind_range_muts = [0]*num_samples + [1]*num_samples + [2]*num_samples + \
                    [3] * num_samples + [4]*num_samples + [5]*num_samples + [6]*num_samples + [7]*num_samples
            
            elif large_or_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
                # The -1 means real distribution
                ind_range_muts = [0]*num_samples + [1]*num_samples + \
                    [2]*num_samples + [3] * num_samples + [-1]*num_samples

            nummuts = [np.random.randint(range_muts[ind_range_muts[i]], range_muts[ind_range_muts[i] + 1])\
                        if ind_range_muts[i] != -1 else -1 for i in range(len(ind_range_muts))]
            return nummuts

        # For testing we use exact number of mutations
        num_muts = [25]*num_samples + [50]*num_samples + [100]*num_samples + [250]*num_samples + [500]*num_samples + [1e3]*num_samples +\
            [5e3]*num_samples + [1e4]*num_samples + [5e4]*num_samples + [1e5]*num_samples
        return num_muts
    
    def __sample_from_sig(self, signature, num_mut, normalize=True):
        sample = signature
        if num_mut > 0 and num_mut < 1e5:  # only do sampling for num_mut < 1e5
            c = torch.distributions.categorical.Categorical(probs=signature)
            sample_shape = torch.Size([num_mut, ])
            samples = c.sample(sample_shape=sample_shape).type(torch.float32)
            n_bins = signature.shape[0]
            sample = torch.histc(samples, bins=n_bins, min=0, max=n_bins - 1)
        if normalize:
            sample = sample/float(num_mut)
        return sample

    def make_random_set(self,
                        split,
                        large_low,
                        num_samples,
                        min_n_signatures=1,
                        max_n_signatures=10,
                        normalize=True):
        """Create a labelled dataset of mutation vectors
        randomly combining the signatures.
        """
        assert(min_n_signatures > 0 and max_n_signatures <= self.total_signatures)

        nummuts = self.__get_nummuts(split, large_low)

        input_batch = torch.empty((len(nummuts), 96))
        label_batch = torch.empty((len(nummuts), self.total_signatures + 1))

        for nummut in tqdm(nummuts):
            # Pick the number of involved signatures
            n_signatures = np.random.randint(
                min_n_signatures, max_n_signatures + 1)

            # Select n_signatures
            signature_ids = torch.randperm(self.total_signatures)[
                :n_signatures]

            # Assign weights randomly
            weights = torch.rand(size=(n_signatures,))
            weights = weights/torch.sum(weights)
            weights[weights > 0.1] = 0  # We want weights larger than 0.1
            weights = weights/torch.sum(weights)
            label = torch.zeros(self.total_signatures).scatter_(
                dim=0, index=signature_ids, src=weights)

            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, label))
            sample = self.__sample_from_sig(signature=signature,
                                                num_mut=int(num_mut),
                                                normalize=normalize)
                
            # Store
            # For the real distribution we say we have more than 1e5 mutations
            nummut = float(nummut) if nummut != -1 else float(np.random.randint(1e5, 1e6))
            input_batch[i, :] = sample
            label_batch[i, :] = torch.cat(label, torch.tensor([float(num_mut)]))
        return input_batch, label_batch

    def make_realistic_set(self,
                           generator_model_path,
                           split,
                           large_low,
                           std = 1.0, 
                           normalize=True):
        """Create a labelled dataset of mutation vectors
        from the generator output weights.
        """
        generator = read_model(generator_model_path, device="cpu")

        nummuts = self.__get_nummuts(split, large_low)

        labels = generator.generate(batch_size, std = std)
        input_batch = torch.empty((batch_size, 96))
        labels_batch = torch.empty((batch_size, self.total_signatures + 1))

        for i, num_mut in tqdm(enumerate(nummuts)):
            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, labels[i]))

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
    

    def make_input(self, labels, split, large_low, normalize=True, seed = 0):
        """Create a labelled dataset of mutation vectors
        from a tensor of labels.
        """
        self.set_all_seeds(seed)
        if split == "train":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
            elif large_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
        elif split == "val":
            if large_low == 'low':
                range_muts = [15, 50, 100, 250, 500, 1000, 5000, 1e4, 1e5]
            elif large_low == 'large':
                range_muts = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
        elif split == "test":
            range_muts = [25, 50, 100, 250, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 1]

        batch_size = (len(range_muts)-1)*labels.shape[0]
        input_batch = torch.empty((batch_size, 96))
        labels_batch = torch.empty((batch_size, self.total_signatures + 1))

        for i in tqdm(range(len(range_muts)-1)):
            for j in range(labels.shape[0]):
                # Compute resulting signature
                signature = torch.einsum("ij,j->i", (self.signatures, labels[j]))

                if split == "test":
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