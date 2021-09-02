import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

class DataLoader:  #TODO(oleguer): Inherit from torch.utils.data.Dataset

    def __init__(self, signatures, batch_size, n_samples=1000, min_n_signatures=1, max_n_signatures=10, seed=None):
        if seed is not None:
            torch.seed = seed
            np.random.seed(seed)
        self.__total_signatures = len(signatures)
        self.signatures = torch.stack(signatures).t()
        self.batch_size = batch_size
        self.n_samples = n_samples
        assert(min_n_signatures > 0 and max_n_signatures <= self.__total_signatures)
        self.min_n_signatures = min_n_signatures  # Inclusive
        self.max_n_signatures = max_n_signatures  # Inclusive

    

    def get_batch(self, normalize=True):
        input_batch = torch.empty((self.batch_size, 96))
        label_batch = torch.empty((self.batch_size, self.__total_signatures + 1))

        for i in range(self.batch_size):
            # Pick the number of involved signatures
            n_signatures = np.random.randint(self.min_n_signatures, self.max_n_signatures + 1)

            # Select n_signatures
            signature_ids = torch.randperm(self.__total_signatures)[:n_signatures]

            # Assign weights randomly
            weights = torch.rand(size=(n_signatures,))
            weights = weights/torch.sum(weights)

            # We want weights larger than 0.1
            for j in range(len(weights)):
                if weights[j]<0.1:
                    weights[j] = 0
            weights = weights/torch.sum(weights)
            label = torch.zeros(self.__total_signatures).scatter_(dim=0, index=signature_ids, src=weights)

            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, label))
            
            # Sample
            if self.n_samples == 0:
                n_samples_1 = np.random.randint(100, 2000)
            c = torch.distributions.categorical.Categorical(probs=signature)
            samples = c.sample(sample_shape=torch.Size([n_samples_1,])).type(torch.float32)
            sample = torch.histc(samples, bins=96, min=0, max=95)
            if normalize:
                sample = sample/float(n_samples_1)
            
            # Store
            input_batch[i, :] = sample
            label_batch[i, :] = torch.cat([label, torch.tensor([n_samples_1])])
        return input_batch, label_batch

    def select_batch(self, training_input, training_label, training_baseline, current_ind=0):
        if current_ind < len(training_input):
            last_ind = current_ind + self.batch_size
        else:
            current_ind = 0
            last_ind = self.batch_size  
        input_batch = training_input[current_ind:last_ind]
        label_batch = training_label[current_ind:last_ind]
        baseline_batch = training_baseline[current_ind:last_ind]

        return input_batch, label_batch, baseline_batch, last_ind


    def make_random_train_set(self, normalize=True):
        batch_size = 150000

        input_batch = torch.empty((batch_size, 96))
        label_batch = torch.empty((batch_size, self.__total_signatures + 1))

        range_muts = [15, 100, 500, 5000, 50000, 500000, 50000000]
        ind_range_muts = [0]*30000 + [1]*30000 + [2]*20000 + [3]*20000 + [4]*25000 + [5]*25000
        for i in range(batch_size):
            # Pick the number of involved signatures
            n_signatures = np.random.randint(self.min_n_signatures, self.max_n_signatures + 1)

            # Select n_signatures
            signature_ids = torch.randperm(self.__total_signatures)[:n_signatures]

            # Assign weights randomly
            weights = torch.rand(size=(n_signatures,))
            weights = weights/torch.sum(weights)

            # We want weights larger than 0.1
            for j in range(len(weights)):
                if weights[j]<0.1:
                    weights[j] = 0
            weights = weights/torch.sum(weights)
            label = torch.zeros(self.__total_signatures).scatter_(dim=0, index=signature_ids, src=weights)

            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, label))
            
            # Sample
            num_mut = np.random.randint(range_muts[ind_range_muts[i]], range_muts[ind_range_muts[i]+1])
            c = torch.distributions.categorical.Categorical(probs=signature)
            samples = c.sample(sample_shape=torch.Size([num_mut,])).type(torch.float32)
            sample = torch.histc(samples, bins=96, min=0, max=95)
            if normalize:
                sample = sample/float(num_mut)
            
            # Store
            input_batch[i, :] = sample
            label_batch[i, :] = torch.cat([label, torch.tensor([num_mut])])
        return input_batch, label_batch