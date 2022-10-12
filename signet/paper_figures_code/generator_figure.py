import logging
import os

import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from signet import DATA, TRAINED_MODELS
from signet.models import Generator
from signet.utilities.io import csv_to_tensor, read_model, tensor_to_csv
from signet.utilities.plotting import plot_correlation_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
real_data_path = os.path.join(DATA, "real_data/sigprofiler_not_norm_PCAWG.csv")
real_data_pd = pd.read_csv(real_data_path)
zero_cols = []
zero_indexes = []

def get_real_correlation_matrix():
    real_data_tensor = csv_to_tensor(real_data_path, device, header=0, index_col=0)
    return plot_correlation_matrix(real_data_tensor, sig_names=real_data_pd.columns[1:])

def get_generator_correlation_matrix():
    generator_path = os.path.join(TRAINED_MODELS, "generator")
    generator = read_model(generator_path, device)

    synt_labels = generator.generate(1000, std=1.0)
    for i in zero_indexes:
        synt_labels[:, i] = 0
    return plot_correlation_matrix(synt_labels[:, :65], sig_names=real_data_pd.columns[1:])

def get_synsiggen_data():
    data_file = "SynSigGen_labels.csv"
    df = pd.read_csv(data_file, index_col=0, header=0)
    # print(real_data_pd.columns[1:])
    # print(df.head)
    # print(df.describe())
    for indx, col in enumerate(real_data_pd.columns[1:]):
        if col not in df.columns:
            df.insert(indx, col, 0)
            zero_cols.append(col)
            zero_indexes.append(indx)

    return plot_correlation_matrix(torch.tensor(df.values), sig_names=real_data_pd.columns[1:])


def get_metrics(real, guess):
    real_ = np.nan_to_num(x=real, nan=0.0, posinf=0.0, neginf=0.0)
    guess_ = np.nan_to_num(x=guess, nan=0.0, posinf=0.0, neginf=0.0)
    mask_ = np.abs(real_) > 0.05

    plt.plot((real_ - guess_)[:, 6])
    plt.show()

    diff = np.abs(real_ - guess_) * mask_

    print("diff count:", np.mean(diff > 0.01))
    print("diff mean:", np.mean(diff))
    return diff


if __name__ == "__main__":
    real = get_real_correlation_matrix()
    synsiggen = get_synsiggen_data()
    signetgen = get_generator_correlation_matrix()

    # for col in zero_cols:
    #     signetgen[col].values[:] = 0

    signetgen_metrics = get_metrics(real, signetgen)
    synsiggen_metrics = get_metrics(real, synsiggen)

    plt.matshow(signetgen_metrics)
    plt.show()

    plt.matshow(synsiggen_metrics)
    plt.show()