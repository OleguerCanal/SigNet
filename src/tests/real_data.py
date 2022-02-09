import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import csv_to_tensor, read_signatures
from utilities.plotting import plot_reconstruction
from utilities.normalize_data import normalize_data
from utilities.metrics import get_cosine_similarity

if __name__=="__main__":
    data_folder = "../../data/"


    # PCAWG DATA:
    inputs = data_folder + "real_data/PCAWG_data.csv"
    labels = data_folder + "real_data/sigprofiler_normalized_PCAWG.csv"

    signatures = read_signatures(data_folder + "data.xlsx")
    inputs = csv_to_tensor(file=inputs, header=0, index_col=0)
    labels = csv_to_tensor(labels, header=0, index_col=0)
    labels = labels/torch.sum(labels, axis=1).reshape(-1, 1)
    labels = torch.cat([labels, torch.zeros(labels.size(0), 7).to(labels)], dim=1)
    num_muts_PCAWG = torch.sum(inputs, axis=1).detach().numpy()
    inputs = normalize_data(inputs,
                            opportunities_name_or_path="genome")
    inputs = inputs/torch.sum(inputs, axis=1).view(-1, 1)

    # reconstruction = torch.einsum("ij,bj->bi", (signatures, torch.tensor(labels)))
    # cosines = get_cosine_similarity(reconstruction, inputs, dim=1)
    # for cos in cosines[:10]:
    #     print(cos.item())


    # plot_reconstruction(input=inputs,
    #                     weight_guess=labels,
    #                     signatures=signatures,
    #                     ind_list=[0,1,2,3],
    #                     plot_path="")


    # MC3 DATA:
    inputs = data_folder + "analysis_MC3/MC3_mut_counts.csv"
    labels = data_folder + "analysis_MC3/weight_guesses.csv"

    signatures = read_signatures(data_folder + "data.xlsx")
    inputs = csv_to_tensor(file=inputs, header=0, index_col=0)
    labels = csv_to_tensor(labels, header=0, index_col=0)
    num_muts_MC3 = torch.sum(inputs, axis=1).detach().numpy()
    print(num_muts_MC3)
    print(num_muts_PCAWG)
    inputs = normalize_data(inputs,
                            opportunities_name_or_path="exome")
    inputs = inputs/torch.sum(inputs, axis=1).view(-1, 1)

    # reconstruction = torch.einsum("ij,bj->bi", (signatures, torch.tensor(labels)))
    # cosines = get_cosine_similarity(reconstruction, inputs, dim=1)
    # for cos in cosines[:97]:
    #     print(cos.item())


    # plot_reconstruction(input=inputs,
    #                     weight_guess=labels,
    #                     signatures=signatures,
    #                     ind_list=[91,92,93,94,95,96],
    #                     plot_path="")


    # GTeX Data:
    inputs = csv_to_tensor(file=data_folder + "case_study_GTeX/data_by_donor/all_donors_input.csv", header=None, index_col=None)
    num_muts_GTeX = torch.sum(inputs, axis=1).detach().numpy()

    # Moore Data:
    inputs = csv_to_tensor(file=data_folder + "case_study_Moore/mut_counts_by_sample.csv", header=0, index_col=0)
    num_muts_Moore = torch.sum(inputs, axis=1).detach().numpy()


    plt.hist(np.log(num_muts_MC3), alpha=0.6, label='MC3')
    plt.hist(np.log(num_muts_PCAWG), alpha=0.6, label='PCAWG')
    plt.hist(np.log(num_muts_GTeX), alpha=0.6, label='GTeX')
    plt.hist(np.log(num_muts_Moore), alpha=0.6, label='Moore')
    plt.ylabel("Number of samples")
    plt.xlabel('log(N)')
    plt.legend()
    plt.show()