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
        input_batch = torch.empty(self.batch_size, 96)
        label_batch = torch.empty(self.batch_size, self.__total_signatures)

        for i in range(self.batch_size):
            # Pick the number of involved signatures
            n_signatures = np.random.randint(self.min_n_signatures, self.max_n_signatures + 1)

            # Select n_signatures
            signature_ids = torch.randperm(self.__total_signatures)[:n_signatures]

            # Assign weights randomly
            weights = torch.rand(size=(n_signatures,)) + 1e-6
            weights = weights/torch.sum(weights)
            label = torch.zeros(self.__total_signatures).scatter_(dim=0, index=signature_ids, src=weights)

            # Compute resulting signature
            signature = torch.einsum("ij,j->i", (self.signatures, label))
            
            # Sample
            c = torch.distributions.categorical.Categorical(probs=signature)
            samples = c.sample(sample_shape=torch.Size([self.n_samples,])).type(torch.float32)
            sample = torch.histc(samples, bins=96, min=0, max=95)
            if normalize:
                sample = sample/float(self.n_samples)
            
            # Store
            input_batch[i, :] = sample
            label_batch[i, :] = label
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