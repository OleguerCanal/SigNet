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
    plot_correlation_matrix(real_data_tensor, sig_names=real_data_pd.columns[1:])

def get_generator_correlation_matrix():
    # generator = Generator()
    # for name, param in generator.named_parameters():
    #     print(name, param.shape)
    generator = read_model(os.path.join(TRAINED_MODELS, "generator_FUCKED"), device)
    synt_labels = generator.generate(1000, std=2.0)
    plot_correlation_matrix(synt_labels, sig_names=list(range(72)))

if __name__ == "__main__":
    # get_real_correlation_matrix()
    get_generator_correlation_matrix()