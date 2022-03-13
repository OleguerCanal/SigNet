import os
import sys

import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utilities.plotting import plot_correlation_matrix
from utilities.io import csv_to_tensor, read_model, read_signatures, sort_signatures

def read_real_data():
    inputs = data_folder + "real_data/PCAWG_data.csv"
    labels = data_folder + "real_data/sigprofiler_not_norm_PCAWG.csv"
    inputs = csv_to_tensor(file=inputs, header=0, index_col=0)
    labels = csv_to_tensor(labels, header=0, index_col=0)
    labels = labels/torch.sum(labels, axis=1).reshape(-1, 1)
    labels = torch.cat([labels, torch.zeros(labels.size(0), 7).to(labels)], dim=1)

    nummut = torch.sum(inputs, dim=1)
    # inputs = normalize_data(inputs,
                            # opportunities_name_or_path="../../data/real_data/3mer_WG_hg37.txt")
                            # opportunities_name_or_path="../../data/real_data/abundances_trinucleotides.txt")
                            # opportunities_name_or_path="../../data/real_data/norm_38.txt")
                            # opportunities_name_or_path="../../data/real_data/new_norm.txt")
    inputs = inputs/torch.sum(inputs, axis=1).view(-1, 1)
    return inputs, labels, nummut

if __name__=="__main__":
    data_folder = "../../../data/"

    real_inputs, real_labels, real_nummut = read_real_data()
    generator_vae = read_model("../../../trained_models/exp_gan/generator_vae")
    generator_gan = read_model("../../../trained_models/exp_gan/gan_pretrained_generator")
    noise = torch.randn((2000, 40))
    synt_labels_gan = generator_gan(noise)
    synt_labels_vae = generator_vae.decode(noise)

    for l in synt_labels_vae[0:10]:
        print(l)
    # print(synt_labels[0])

    # Correlation matrices
    signatures = sort_signatures(file=data_folder + "data.xlsx",
                                 mutation_type_order=data_folder + "mutation_type_order.xlsx")
    plot_correlation_matrix(data=real_labels, signatures=signatures)
    plot_correlation_matrix(data=synt_labels_vae, signatures=signatures)
    plot_correlation_matrix(data=synt_labels_gan, signatures=signatures)

