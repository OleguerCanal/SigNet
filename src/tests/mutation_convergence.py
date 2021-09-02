import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm

if __name__=="__main__":
    signatures_data = pd.read_excel("../../data/data.xlsx")
    signatures = [torch.tensor(signatures_data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:72]
    total_signatures = len(signatures)
    signatures = torch.stack(signatures).t()

    n_mutations = [25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
    # n_mutations = [10e3, 20e3, 50e3, 100e3, 500e3]
    # n_mutations += [1e6, 5e6, 10e6, 50e6]

    errors = []

    for n_mut in tqdm(n_mutations):
        n_mut = int(n_mut)
        n_signatures = 5
        signature_ids = torch.randperm(total_signatures)[:n_signatures]
        weights = torch.rand(size=(n_signatures,))
        weights = weights/torch.sum(weights)
        
        for j in range(len(weights)):
            if weights[j]<0.1:
                weights[j] = 0
        weights = weights/torch.sum(weights)
        label = torch.zeros(total_signatures).scatter_(dim=0, index=signature_ids, src=weights)
        signature = torch.einsum("ij,j->i", (signatures, label))
        c = torch.distributions.categorical.Categorical(probs=signature)
        samples = c.sample(sample_shape=torch.Size([n_mut,])).type(torch.float32)
        sample = torch.histc(samples, bins=96, min=0, max=95)/float(n_mut)
        errors.append(torch.nn.MSELoss()(sample, signature).item())

    plt.plot(n_mutations, errors)
    plt.ylabel("MSE")
    plt.xlabel("Num mutations")
    plt.show()