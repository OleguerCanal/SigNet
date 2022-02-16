import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import csv_to_tensor, read_signatures
from utilities.plotting import plot_reconstruction
from utilities.io import read_signatures, read_test_data, read_model, csv_to_tensor
from utilities.plotting import plot_reconstruction
from utilities.normalize_data import normalize_data
from utilities.metrics import get_cosine_similarity
from modules.combined_finetuner import CombinedFinetuner


def read_real_data():
    inputs = data_folder + "real_data/PCAWG_data.csv"
    labels = data_folder + "real_data/sigprofiler_normalized_PCAWG.csv"

    inputs = csv_to_tensor(file=inputs, header=0, index_col=0)
    labels = csv_to_tensor(labels, header=0, index_col=0)
    labels = labels/torch.sum(labels, axis=1).reshape(-1, 1)
    labels = torch.cat([labels, torch.zeros(labels.size(0), 7).to(labels)], dim=1)

    nummut = torch.sum(inputs, dim=1)
    inputs = normalize_data(inputs,
                            opportunities_name_or_path="../../data/real_data/new_norm.txt")
    inputs = inputs/torch.sum(inputs, axis=1).view(-1, 1)
    return inputs, labels, nummut


def read_synt_data():
    experiment_id = "exp_generator"
    test_id = "test_generator"
    input_batch, label_batch = read_test_data("cpu", experiment_id, test_id, data_folder="../../data")
    return input_batch, label_batch[:, :-1], label_batch[:, -1]

if __name__=="__main__":
    data_folder = "../../data/"

    signatures = read_signatures(data_folder + "data.xlsx")
    print(signatures)
    signatures_norm = normalize_data(signatures.t(), "genome").t()
    signatures_norm = signatures_norm / torch.sum(signatures_norm, dim=0)#.reshape(-1, 1)
    print(signatures_norm)

    plt.bar(range(96), signatures[:,0], width=0.4)
    plt.bar(np.array(range(96))+0.4, signatures_norm[:,0], width=0.4)
    plt.show()

    input_file_path = "../../data/case_study/PCAWG/mappability/pancan_mutations_counts_CRG36.bed"
    weights_file_path = "../../data/case_study/PCAWG/mappability/SigNet_output/weight_guesses.csv"
    opportunities = "../../data/real_data/abundances_WG_CRG36.txt"
    
    inputs = csv_to_tensor(file=input_file_path, header=0, index_col=0)
    labels = csv_to_tensor(weights_file_path, header=0, index_col=0)

    # plt.bar(range(96), inputs[2,:], width=0.4)
    # plt.show()
    # plt.close()

    inputs = normalize_data(inputs, opportunities)
    inputs = inputs / torch.sum(inputs, dim=1).reshape(-1, 1)
  
    # print(inputs)
    # print(labels)
    # plot_reconstruction(input=inputs,
    #                     weight_guess=labels,
    #                     signatures=signatures,
    #                     ind_list=[0,1,2,3],
    #                     plot_path="")

    # input_file_path = "../../data/case_study/PCAWG/PCAWG_Claudia_all/mutation_counts/pancan_mutations.bed"
    # weights_file_path = "../../data/case_study/PCAWG/SigNet/WGS/weight_guesses.csv"
    # opportunities = "genome"
    
    # input_file_path = "../../../PCAWG/PCAWG_input/WGS_PCAWG.96.csv"
    # weights_file_path = "../../../PCAWG/PCAWG_sigProfiler_SBS_signatures.csv"
    # sigs_file_path = "../../../PCAWG/sigProfiler_SBS_signatures_2019_05_22.csv"

    # inputs = csv_to_tensor(file=input_file_path, header=0, index_col=[0,1])
    # inputs = inputs.transpose(1,0)
    # print(inputs)
    # labels = csv_to_tensor(weights_file_path, header=0, index_col=0)
    # print(labels)
    # sigs = csv_to_tensor(sigs_file_path, header=0, index_col=[0,1, -1, -2])
    # print(sigs)

    # labels = torch.cat([labels, torch.zeros(labels.size(0), 7).to(labels)], dim=1)

    # inputs = normalize_data(inputs, opportunities)
    # inputs = inputs / torch.sum(inputs, dim=1).reshape(-1, 1)
  
    # print(inputs)
    # print(labels)
    # plot_reconstruction(input=inputs,
    #                     weight_guess=labels,
    #                     signatures=sigs,
    #                     ind_list=[0,1,2,3,100,200,300],
    #                     plot_path="")


    # reconstruction = torch.einsum("ij,bj->bi", (sigs, labels))
    # cosines = get_cosine_similarity(reconstruction, inputs, dim=1)
    # for cos in cosines[:10]:
    #     print(cos.item())