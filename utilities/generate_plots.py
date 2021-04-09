import os
import time
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline import SignatureFinder
from model import SignatureNet
from utilities.dataloader import DataLoader
from utilities.metrics import *
from utilities.plotting import plot_signature, plot_confusion_matrix, plot_weights, plot_weights_comparison, plot_weights_comparison_deconstructSigs
from model_tester import ModelTester


if __name__ == "__main__":
    # Model params
    experiment_id = "learn_error_6"
    num_classes = 72
    # Generate data
    data = pd.read_excel("../data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                 for i in range(2, 74)][:num_classes]
    # Instantiate model and do predictions
    model = SignatureNet(signatures=signatures,
                          num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join("../models", experiment_id)))
    model.eval()



    # Test set: 
    input_batch = torch.tensor(pd.read_csv("../data/test_input_w01.csv", header=None).values, dtype=torch.float)
    label_mut_batch = torch.tensor(pd.read_csv("../data/test_label_w01.csv", header=None).values, dtype=torch.float)
    label_batch = label_mut_batch[:,:num_classes]
    num_mut = torch.reshape(label_mut_batch[:,num_classes], (list(label_mut_batch.size())[0],1))
    baseline_batch = torch.tensor(pd.read_csv("../data/test_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    deconstructSigs_batch = torch.tensor(pd.read_csv("../data/deconstructSigs_test_w01.csv", header=None).values, dtype=torch.float)
    guessed_error = model(baseline_batch, num_mut)

            # Plot comparison real, deconstructSigs and guessed
    plot_weights_comparison_deconstructSigs(label_batch[15,:].detach().numpy(), deconstructSigs_batch[15,:].detach().numpy(), baseline_batch[15,:].detach().numpy(), guessed_error[15,:].detach().numpy(), list(data.columns)[2:]) #1971 mutations
    plot_weights_comparison_deconstructSigs(label_batch[22,:].detach().numpy(), deconstructSigs_batch[22,:].detach().numpy(), baseline_batch[22,:].detach().numpy(), guessed_error[22,:].detach().numpy(), list(data.columns)[2:]) # sig 5, 739 mut
    plot_weights_comparison_deconstructSigs(label_batch[9986,:].detach().numpy(), deconstructSigs_batch[9986,:].detach().numpy(), baseline_batch[9986,:].detach().numpy(), guessed_error[9986,:].detach().numpy(), list(data.columns)[2:]) #17 mutations


    # MC3 plots: Cancer type specific.
    num_mut = torch.tensor(pd.read_csv("../data/MC3_data_number_mut.csv", header=None).values, dtype=torch.float)
    num_mut = torch.reshape(num_mut, (list(num_mut.size())[0],1))
    baseline_batch = torch.tensor(pd.read_csv("../data/MC3_data_baseline_JS.csv", header=None).values, dtype=torch.float)
    deconstructSigs_batch = torch.tensor(pd.read_csv("../data/MC3_data_deconstructSigs.csv", header=None).values, dtype=torch.float)
    guessed_error = model(baseline_batch, num_mut)

            # Plot signatures comparison between deconstructSigs
    plot_weights_comparison(deconstructSigs_batch[1,:].detach().numpy(),baseline_batch[1,:].detach().numpy(), guessed_error[1,:].detach().numpy(), list(data.columns)[2:]) # Sig 10b
    plot_weights_comparison(deconstructSigs_batch[27,:].detach().numpy(),baseline_batch[27,:].detach().numpy(), guessed_error[27,:].detach().numpy(), list(data.columns)[2:]) # Sig 87
    plot_weights_comparison(deconstructSigs_batch[31,:].detach().numpy(),baseline_batch[31,:].detach().numpy(), guessed_error[31,:].detach().numpy(), list(data.columns)[2:]) #Sig 87 and 10b


    # MC3 plots: Samples specific, ACC cancer type.
    num_mut = torch.tensor(pd.read_csv("../data/MC3_ACC_num_mut.csv", header=None).values, dtype=torch.float)
    num_mut = torch.reshape(num_mut, (list(num_mut.size())[0],1))
    baseline_batch = torch.tensor(pd.read_csv("../data/MC3_ACC_data_baseline_JS.csv", header=None).values, dtype=torch.float)
    deconstructSigs_batch = torch.tensor(pd.read_csv("../data/MC3_ACC_deconstructSigs.csv", header=None).values, dtype=torch.float)
    guessed_error = model(baseline_batch, num_mut)

            # Plot signatures comparison between deconstructSigs
    plot_weights_comparison(deconstructSigs_batch[1,:].detach().numpy(), baseline_batch[1,:].detach().numpy(), guessed_error[1,:].detach().numpy(), list(data.columns)[2:]) # Sig 87 and 10b, 15 mutations
    plot_weights_comparison(deconstructSigs_batch[16,:].detach().numpy(),baseline_batch[16,:].detach().numpy(), guessed_error[16,:].detach().numpy(), list(data.columns)[2:]) # A lot of sig 87 and ont sig 1, 34 mutations
    plot_weights_comparison(deconstructSigs_batch[48,:].detach().numpy(),baseline_batch[48,:].detach().numpy(), guessed_error[48,:].detach().numpy(), list(data.columns)[2:]) # A lot of sig 10b, 28 mutations
    plot_weights_comparison(deconstructSigs_batch[2,:].detach().numpy(),baseline_batch[2,:].detach().numpy(), guessed_error[2,:].detach().numpy(), list(data.columns)[2:]) # Sig 87 and 2231 mutations
