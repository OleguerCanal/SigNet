import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import csv_to_tensor, read_signatures
from utilities.plotting import plot_reconstruction
from utilities.normalize_data import normalize_data
from utilities.metrics import get_cosine_similarity

if __name__=="__main__":
    data_folder = "../../data/"
    inputs = data_folder + "real_data/PCAWG_data.csv"
    labels = data_folder + "real_data/sigprofiler_normalized_PCAWG.csv"

    signatures = read_signatures(data_folder + "data.xlsx")
    inputs = csv_to_tensor(file=inputs, header=0, index_col=0)
    labels = csv_to_tensor(labels, header=0, index_col=0)
    labels = labels/torch.sum(labels, axis=1).reshape(-1, 1)
    labels = torch.cat([labels, torch.zeros(labels.size(0), 7).to(labels)], dim=1)

    inputs = normalize_data(inputs,
                            opportunities_name_or_path="../../data/real_data/new_norm.txt")
    inputs = inputs/torch.sum(inputs, axis=1).view(-1, 1)

    reconstruction = torch.einsum("ij,bj->bi", (signatures, torch.tensor(labels)))
    cosines = get_cosine_similarity(reconstruction, inputs, dim=1)
    for cos in cosines[:10]:
        print(cos.item())


    plot_reconstruction(input=inputs,
                        weight_guess=labels,
                        signatures=signatures,
                        ind_list=[1],
                        plot_path="")

    