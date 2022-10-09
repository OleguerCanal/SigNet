import logging
import os

import pandas as pd
import torch

from signet import DATA, TRAINED_MODELS
from signet.models import Generator
from signet.utilities.io import csv_to_tensor, read_model, tensor_to_csv
from signet.utilities.plotting import plot_correlation_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
real_data_path = os.path.join(DATA, "real_data/sigprofiler_not_norm_PCAWG.csv")
real_data_pd = pd.read_csv(real_data_path)


def get_real_correlation_matrix():
    real_data_tensor = csv_to_tensor(real_data_path, device, header=0, index_col=0)
    print(real_data_tensor.shape)
    plot_correlation_matrix(real_data_tensor, sig_names=real_data_pd.columns[1:])

def get_generator_correlation_matrix():
    generator_path = os.path.join(TRAINED_MODELS, "generator")
    generator = read_model(generator_path, device)

    synt_labels = generator.generate(1000, std=1.0)
    print(synt_labels.shape)
    plot_correlation_matrix(synt_labels[:, :65], sig_names=real_data_pd.columns[1:])

def get_synsiggen_data():
    data_file = "/home/oleguer/projects/signatures-net/signet/paper_figures_code/realistic_train_label.csv"
    synsiggen_data_tensor = csv_to_tensor(data_file, device)
    plot_correlation_matrix(synsiggen_data_tensor[:, :65], sig_names=real_data_pd.columns[1:])


if __name__ == "__main__":
    # get_real_correlation_matrix()
    # get_generator_correlation_matrix()
    get_synsiggen_data()