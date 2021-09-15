import sys
import os
import numpy as np

import pandas as pd
import torch
from torch.serialization import default_restore_location

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_real_data, read_signatures, write_data
from models.finetuner import baseline_guess_to_finetuner_guess
from models.yapsa_inspired_baseline import YapsaInspiredBaseline

def sample_from_sig(signature, num_mut):
    c = torch.distributions.categorical.Categorical(probs=signature)
    samples = c.sample(sample_shape=torch.Size([num_mut,])).type(torch.float32)
    sample = torch.histc(samples, bins=96, min=0, max=95)
    sample = sample/float(num_mut)
    return sample

# Parameters
num_classes = 72
device = "cpu"
experiment_id = "exp_real_data"

signatures = read_signatures("../../data/data.xlsx")

real_input, real_num_mut = read_real_data(device=device,
                           experiment_id=experiment_id,
                           data_folder="../../data")

# Baseline:
baseline = YapsaInspiredBaseline(signatures)
baseline_guess = baseline.get_weights_batch(real_input) 

# Amplification:
# label = amplification(baseline_guess) sort of thing

# Sampling:
signature = torch.einsum("ij,j->i", (signatures, label))
range_muts = [15, 50, 75, 100, 150, 250, 500, 1e3, 1e4, 1e5, 1e6]
data_input = torch.empty((signature.shape[0]*(len(range_muts)-1), 96))
for i in range(len(range_muts)-1):
    for j in range(signature.shape[0]):
        num_mut = np.random.randint(range_muts[i], range_muts[i+1])
        data_input[j+i*signature.shape[0],:] = sample_from_sig(signature[j,:], num_mut)
    data_label = torch.cat((label,label),0)

# Write results:
write_data(data_label, "../../data/%s/data_label.csv"%experiment_id)
write_data(data_input, "../../data/%s/data_input.csv"%experiment_id)

# Maybe already save the training, validation and test set?