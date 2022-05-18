
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from modules.combined_finetuner import CombinedFinetuner
from tests.real_data import read_finetuner
from trainers.finetuner_crossvalidation import read_data_and_partitions
from utilities.data_generator import DataGenerator
from utilities.io import csv_to_pandas, csv_to_tensor, read_signatures, sort_signatures
from utilities.metrics import get_MSE
from utilities.oversampler import CancerTypeOverSampler
from utilities.plotting import get_correlation_matrix, plot_correlation_matrix


def bootstrap_std(set, bootstrap_size, N_bootstrap, signatures):
    corr_list = []
    for _ in range(N_bootstrap):
        ind = np.random.choice(range(bootstrap_size),test_label.size(0))
        bootstrap_set = set[ind,:]
        corr_list.append(get_correlation_matrix(bootstrap_set, signatures))
    return np.dstack(([d for d in corr_list])).std(axis=2)



data_folder = "../../data/"
experiment_id = "oversample_crossval"
k_tot = 10

# Real data
real_data = csv_to_tensor("../../data/real_data/sigprofiler_not_norm_PCAWG.csv",
                              device='cpu', header=0, index_col=0)
real_data = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)

# Create partitions
lst_weights, lst_ctype = read_data_and_partitions(k_tot)

# Create inputs associated to the labels:
signatures = read_signatures(
    "../../data/data.xlsx", mutation_type_order="../../data/mutation_type_order.xlsx")
data_generator = DataGenerator(signatures=signatures,
                                seed=None,
                                shuffle=True)

models_path = "../../trained_models/" + experiment_id

N_oversample_list = [1, 5, 10, 20, 30, 33]

mse_oversample = []
mse_oversample_001 = []
mse_oversample_005 = []
mse_oversample_01 = []
for N_oversample in N_oversample_list:
    # Oversample each set to have the same number of samples for each cancer type
    oversampler = CancerTypeOverSampler(lst_weights[0], lst_ctype[0])
    test_set = oversampler.get_N_oversampled_set(N_oversample)

    # Create pairs input-label
    print("Creating train, val and test data")
    test_input, test_label = data_generator.make_input(test_set, "test", "", normalize=True)

    # Run Baseline
    print("Running Baseline")
    signatures = read_signatures("../../data/data.xlsx")
    sf = Baseline(signatures)
    test_baseline = sf.get_weights_batch(test_input, n_workers=2)

    # Apply model to test set
    finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "/finetuner_low_crossval_0_" + str(N_oversample),
                                    large_mum_mut_dir=models_path + "/finetuner_large_crossval_0_" + str(N_oversample))
    test_guess = finetuner(mutation_dist=test_input,
                            baseline_guess=test_baseline,
                            num_mut=test_label[:, -1].view(-1, 1))

    signatures = sort_signatures("../../data/data.xlsx")
    
    # Bootstrap std
    # std_bootstrap = np.nan_to_num(bootstrap_std(test_label[:,:-1], 1000, signatures))
    std_bootstrap = np.nan_to_num(bootstrap_std(real_data, test_label.size(0), 1000, signatures))

    test_guess_corr = get_correlation_matrix(test_guess, signatures, plot=False)
    test_label_corr = get_correlation_matrix(real_data, signatures, plot=False)
    # test_label_corr = get_correlation_matrix(test_label[:, :-1], signatures, plot=False)
    dif = (torch.nan_to_num(torch.Tensor(test_guess_corr.values))-torch.nan_to_num(torch.Tensor(test_label_corr.values)))**2

    fig = plt.figure()
    sn.heatmap(dif, annot=False)

    fig = plt.figure()
    sn.heatmap(std_bootstrap, annot=False)
    plt.show()

    mse_oversample.append(torch.sum((torch.nan_to_num(torch.Tensor(test_guess_corr.values))-torch.nan_to_num(torch.Tensor(test_label_corr.values)))**2/std_bootstrap).item())

    test_guess[test_guess<0.01] = 0
    test_label[test_label<0.01] = 0
    test_guess_corr = get_correlation_matrix(test_guess, signatures, plot=False)
    test_label_corr = get_correlation_matrix(test_label[:, :-1], signatures, plot=False)
    mse_oversample_001.append(get_MSE(torch.nan_to_num(torch.Tensor(test_guess_corr.values)), torch.nan_to_num(torch.Tensor(test_label_corr.values))).item())

    test_guess[test_guess<0.05] = 0
    test_label[test_label<0.05] = 0
    test_guess_corr = get_correlation_matrix(test_guess, signatures, plot=False)
    test_label_corr = get_correlation_matrix(test_label[:, :-1], signatures, plot=False)
    mse_oversample_005.append(get_MSE(torch.nan_to_num(torch.Tensor(test_guess_corr.values)), torch.nan_to_num(torch.Tensor(test_label_corr.values))).item())

    test_guess[test_guess<0.1] = 0
    test_label[test_label<0.1] = 0
    test_guess_corr = get_correlation_matrix(test_guess, signatures, plot=False)
    test_label_corr = get_correlation_matrix(test_label[:, :-1], signatures, plot=False)
    mse_oversample_01.append(get_MSE(torch.nan_to_num(torch.Tensor(test_guess_corr.values)), torch.nan_to_num(torch.Tensor(test_label_corr.values))).item())
    # plt.show()
    # plt.close()

print(mse_oversample)
plt.plot(N_oversample_list, mse_oversample,'--o', label = 'No threshold')
plt.plot(N_oversample_list, mse_oversample_001,'--o', label = 'Threshold 0.01')
plt.plot(N_oversample_list, mse_oversample_005,'--o', label = 'Threshold 0.05')
plt.plot(N_oversample_list, mse_oversample_01,'--o', label = 'Threshold 0.1')
plt.xlabel("Oversample")
plt.ylabel("MSE")
plt.legend()
plt.show()