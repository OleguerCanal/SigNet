import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signet import DATA
from models import Baseline
from signet.utilities.data_generator import DataGenerator
from signet.utilities.data_partitions import DataPartitions
from signet.utilities.io import csv_to_pandas, read_signatures
from signet.utilities.oversampler import CancerTypeOverSampler


def read_data_final_finetuner(device, data_id, data_folder=DATA, network_type="low"):
    '''
    Read all real data, oversample and generate samples with different numbers of mutations
    to train the final finetuner.
    '''
    data_folder = data_folder + data_id
    real_data = csv_to_pandas(data_folder + "/sigprofiler_not_norm_PCAWG.csv",
                                    device=device, header=0, index_col=0,
                                    type_df=data_folder + "/PCAWG_sigProfiler_SBS_signatures_in_samples_v3.csv")
    ctypes = torch.tensor(real_data.values[:,-1], dtype=torch.float)
    real_data = torch.tensor(real_data.values[:,:-1], dtype=torch.float)
    real_data = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)
    oversampler = CancerTypeOverSampler(real_data, ctypes)
    oversampled_weights = oversampler.get_N_oversampled_set(N_samples=1)

    # Create inputs associated to the labels:
    signatures = read_signatures(
        DATA + "data.xlsx",
        mutation_type_order=DATA + "mutation_type_order.xlsx")
    data_generator = DataGenerator(signatures=signatures,
                                   seed=None,
                                   shuffle=True)

    train_input, train_label = data_generator.make_input(oversampled_weights, "train", network_type, normalize=True)
    
    # Run Baseline
    sf = Baseline(signatures)
    train_baseline = sf.get_weights_batch(train_input, n_workers=2)
    
    train_data = DataPartitions(inputs=train_input.float().to(device),
                                prev_guess=train_baseline.float().to(device),
                                labels=train_label.float().to(device))
    val_data = train_data
    return train_data, val_data