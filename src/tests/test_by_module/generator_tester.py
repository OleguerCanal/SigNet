import os
import sys

import torch
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utilities.plotting import plot_correlation_matrix, plot_histograms
from utilities.metrics import sets_distances, get_distances_metrics
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
    print("ok")

    generator_vae = read_model("../../../trained_models/fig_generator/oversampled_dotdist_largelatent_generator_5_120_3")
    # generator_vae = read_model("../../../trained_models/exp_real_data/generator_perturbed")
    noise = torch.randn((2000, 120))
    synt_labels_vae = generator_vae.decode(noise)
    # synt_labels_gan = generator_gan(noise)

    real_dists, fake_dists = sets_distances(real_labels, synt_labels_vae)


    print("Real distances")
    real_dists_metrics = get_distances_metrics(real_dists)
    pprint(real_dists_metrics)
    
    print("Fake distances")
    pprint(get_distances_metrics(fake_dists))

    data_dict = {
        "fake": fake_dists,
        "real": real_dists
    }
    plot_histograms(data_dict)

    # Correlation matrices
    # signatures = sort_signatures(file=data_folder + "data.xlsx",
    #                              mutation_type_order=data_folder + "mutation_type_order.xlsx")
    # plot_correlation_matrix(data=real_labels, signatures=signatures)
    # plot_correlation_matrix(data=synt_labels_vae, signatures=signatures)
    # plot_correlation_matrix(data=synt_labels_gan, signatures=signatures)

