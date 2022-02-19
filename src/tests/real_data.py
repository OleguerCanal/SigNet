import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import csv_to_tensor, read_signatures
from utilities.io import read_signatures, read_test_data, read_model, csv_to_tensor
from utilities.plotting import plot_reconstruction, plot_bars
from utilities.normalize_data import normalize_data
from utilities.metrics import get_cosine_similarity
from modules.combined_finetuner import CombinedFinetuner
from models.baseline import Baseline

def read_real_data():
    inputs = data_folder + "real_data/PCAWG_data.csv"
    labels = data_folder + "real_data/sigprofiler_not_norm_PCAWG.csv"
    inputs = csv_to_tensor(file=inputs, header=0, index_col=0)
    labels = csv_to_tensor(labels, header=0, index_col=0)
    labels = labels/torch.sum(labels, axis=1).reshape(-1, 1)
    labels = torch.cat([labels, torch.zeros(labels.size(0), 7).to(labels)], dim=1)

    nummut = torch.sum(inputs, dim=1)
    inputs = normalize_data(inputs,
                            opportunities_name_or_path="../../data/real_data/3mer_WG_hg37.txt")
                            # opportunities_name_or_path="../../data/real_data/abundances_trinucleotides.txt")
                            # opportunities_name_or_path="../../data/real_data/norm_38.txt")
                            # opportunities_name_or_path="../../data/real_data/new_norm.txt")
    inputs = inputs/torch.sum(inputs, axis=1).view(-1, 1)

    baseline = Baseline(inputs)
    return inputs, baseline, labels, nummut

def read_synt_data():
    input_batch = csv_to_tensor("../../data/exp_not_norm/test_generator_input.csv")
    label_batch = csv_to_tensor("../../data/exp_not_norm/test_generator_input.csv")
    baseline_batch = csv_to_tensor("../../data/exp_not_norm/test_generator_input.csv")
    return input_batch, baseline_batch, label_batch[:, :-1], label_batch[:, -1]

def read_finetuner():
    experiment_id = "exp_not_norm"
    models_path = "../../trained_models/%s/"%experiment_id
    finetuner = CombinedFinetuner(low_mum_mut_dir=models_path + "finetuner_not_norm_low",
                                            large_mum_mut_dir=models_path + "finetuner_not_norm_large")
    return finetuner

def normalize(a, b):
    """Normalize 1 wrt b
    """
    a_mean = torch.mean(a, dim=0)
    b_mean = torch.mean(b, dim=0)
    return (a/a_mean)*b_mean

if __name__=="__main__":
    data_folder = "../../data/"

    real_inputs, real_baseline, real_labels, real_nummut = read_real_data()
    synt_inputs, synt_baseline, synt_labels, synt_nummut = read_synt_data()

    real_inputs_norm = normalize(real_inputs, synt_inputs)

    finetuner = read_finetuner()
    real_guess = finetuner(mutation_dist=real_inputs_norm, baseline_guess=real_baseline, num_mut=real_nummut)
    synt_guess = finetuner(mutation_dist=synt_inputs, baseline_guess=synt_baseline, num_mut=synt_nummut)


    signatures = read_signatures(data_folder + "data.xlsx")
    real_label_rec = torch.einsum("ij,bj->bi", (signatures, torch.tensor(real_labels)))
    real_guess_rec = torch.einsum("ij,bj->bi", (signatures, torch.tensor(real_guess)))
    synt_label_rec = torch.einsum("ij,bj->bi", (signatures, torch.tensor(synt_labels)))
    synt_guess_rec = torch.einsum("ij,bj->bi", (signatures, torch.tensor(synt_guess)))

    data = {
             "synt_labels": synt_labels,
             "synt_guess": synt_guess,
             "real_labels": real_labels,
             "real_guess": real_guess,
             }
    plot_bars(data, max=45)

    data = {
            # "synt_inputs": synt_inputs,
            # "synt_label_rec": synt_label_rec,
            # "synt_guess_rec": synt_guess_rec,
            "real_inputs": real_inputs,
            "real_inputs_norm": real_inputs_norm,
            "real_label_rec": real_label_rec,
            "real_guess_rec": real_guess_rec,
            }
    plot_bars(data)

    