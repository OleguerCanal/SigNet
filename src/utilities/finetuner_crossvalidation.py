import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from modules.combined_finetuner import CombinedFinetuner
from utilities.data_generator import DataGenerator
from utilities.data_partitions import DataPartitions
from trainers.finetuner_trainer import train_finetuner
from utilities.io import csv_to_pandas, read_config, read_signatures
from utilities.metrics import get_classification_metrics
from utilities.oversampler import CancerTypeOverSampler

def read_data_and_partitions(k):
    '''
    Read real data and generate k partitions to apply k-fold crossvalidation.
    '''
    real_data = csv_to_pandas("../../data/real_data/sigprofiler_not_norm_PCAWG.csv",
                            device='cpu', header=0, index_col=0,
                            type_df="../../data/real_data/PCAWG_sigProfiler_SBS_signatures_in_samples_v3.csv")
    
    num_ctypes = real_data['cancer_type'][-1]+1

    #Shuffle samples inside the same cancer type
    real_data = real_data.groupby('cancer_type').sample(frac=1, random_state=0)
    num_samples_ctype = real_data.groupby('cancer_type', as_index=False).size()    

    # Partition the data in k groups preserving proportion of samples in each cancer type
    group_size = (num_samples_ctype/k).astype('int')

    lst = [pd.DataFrame(columns=real_data.columns) for _ in range(k)]    
    for i in range(k):
        for ctype in range(num_ctypes):
            lst[i].append(real_data[real_data['ctype']==ctype].reset_index(drop=True).loc[range(group_size[ctype]*i, group_size[ctype]*(i+1))])

    # Remove cancer column, normalize each row and add missing signatuers with 0s at the end
    lst_weights = [torch.tensor(df[:, :-1].values, dtype=torch.float) for df in lst]
    lst_weights = [df/torch.sum(df, axis=1).reshape(-1,1) for df in lst_weights]
    lst_weights = [torch.cat([df, torch.zeros(df.size(0), 7).to(df)], dim=1) for df in lst_weights]

    # Cancer type column
    lst_ctype = [torch.tensor(df[:, -1].values, dtype=torch.float) for df in lst]

    return lst_weights, lst_ctype


if __name__ == "__main__":
    
    k = 10  
    network_type = 'low'
    finetuner_config_path = "../configs/finetuner/finetuner_" + network_type

    # Create partitions
    lst_weights, lst_ctype = read_data_and_partitions(k)

    # Oversample each set to have the same number of samples for each cancer type
    oversampled_weights = []                                           
    for i in range(k):
        oversampler = CancerTypeOverSampler(lst_weights[i], lst_ctype[i])
        oversampled_weights.append(oversampler.get_even_set())     

    # Create inputs associated to the labels:
    signatures = read_signatures("../data/data.xlsx", mutation_type_order="../data/mutation_type_order.xlsx")
    data_generator = DataGenerator(signatures=signatures,
                                   seed=None,
                                   shuffle=True)

    # Loop through the partitions
    for i in range(k):

        #Create train, val and test weight sets
        test_weights = oversampled_weights[i]
        if i < k-2:
            val_weights = oversampled_weights[i+1]
            train_weights = oversampled_weights[i+2]
            for j in range(k):
                if j != i and j != i+1 and j != i+2:
                    train_weights.append(oversampled_weights[j])
        else:
            val_weights = oversampled_weights[0]
            train_weights = oversampled_weights[1]
            for j in range(2,k):
                if j != i:
                    train_weights.append(oversampled_weights[j])

        # Create pairs input-label
        train_input, train_label = data_generator.make_input(train_weights, "train", network_type, normalize=True)
        val_input, val_label = data_generator.make_input(val_weights, "val", network_type, normalize=True)
        test_input, test_label = data_generator.make_input(test_weights, "test", network_type, normalize=True)
    
        # Run Baseline
        sf = Baseline(signatures)
        train_baseline = sf.get_weights_batch(train_input, n_workers=2)
        val_baseline = sf.get_weights_batch(val_input, n_workers=2)
        test_baseline = sf.get_weights_batch(test_input, n_workers=2)
        
        # Create DataPartitions
        train_data = DataPartitions(inputs=train_input,
                                    prev_guess=train_baseline,
                                    labels=train_label)
        val_data = DataPartitions(inputs=val_input,
                                prev_guess=val_baseline,
                                labels=val_label)

        # Read config
        config = read_config(path=finetuner_config_path)
        config["model_id"] = "finetuner_crossval_" + str(i)

        # Train finetuner
        score = train_finetuner(config=config)

        # Apply model to test set
        models_path = config["models_dir"]
        finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "finetuner_low",
                                    large_mum_mut_dir=models_path + "finetuner_large")
        finetuner_guess = finetuner(mutation_dist=test_input,
                                    baseline_guess=test_baseline,
                                    num_mut=test_label[:,-1].view(-1, 1))

        # Test model
        num_muts = np.unique(test_label[:,-1].detach().numpy())
        list_of_metrics = ["MAE", "KL", "fpr", "fnr", "accuracy %", "precision %", "sens: tp/p %", "spec: tn/n %"]
        values = np.zeros((k, len(num_muts), len(list_of_metrics)))
        for i, num_mut in enumerate(num_muts):
            indexes = test_label[:, -1] == num_mut
            metrics = get_classification_metrics(label_batch=test_label[indexes, :-1],
                                                prediction_batch=finetuner_guess)
            for metric_index, metric in enumerate(list_of_metrics):
                values[k, i, metric_index] = metrics[metric]

    # Plot final results
    marker_size = 3
    line_width = 0.5

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    axs[0,0].plot(np.log10(num_muts), np.transpose(values[:,:,0]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[0,1].plot(np.log10(num_muts), np.transpose(values[:,:,1]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,0].plot(np.log10(num_muts), np.transpose(values[:,:,2]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,1].plot(np.log10(num_muts), np.transpose(values[:,:,3]), marker='o',linewidth=line_width, markersize=marker_size)
    plt.show()
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    axs[0,0].plot(np.log10(num_muts), np.transpose(values[:,:,4]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[0,1].plot(np.log10(num_muts), np.transpose(values[:,:,5]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,0].plot(np.log10(num_muts), np.transpose(values[:,:,6]), marker='o',linewidth=line_width, markersize=marker_size)
    axs[1,1].plot(np.log10(num_muts), np.transpose(values[:,:,7]), marker='o',linewidth=line_width, markersize=marker_size)
    plt.show()
    plt.close()
