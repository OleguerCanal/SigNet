import time

import copy
import random

import pandas as pd
import torch

from signet import DATA
from signet.modules.signet_module import SigNet
from signet.utilities.io import read_methods_guesses, tensor_to_csv
from signet.utilities.plotting import final_plot_all_metrics_vs_mutations, final_plot_interval_metrics_vs_mutations, plot_metric_vs_mutations_classifier, plot_time_vs_mutations

import musical

def subsample(n, inputs, labels):
    indexes = list(range(inputs.shape[0]))
    random.shuffle(indexes)
    ind_list = indexes[:n]
    inputs = inputs.iloc[ind_list]
    labels = labels.iloc[ind_list]
    return inputs, labels

def get_musical_guess(inputs, signatures):
    musical_input = copy.deepcopy(inputs).transpose()
    musical_signatures = copy.deepcopy(signatures)
    musical_signatures = musical_signatures.set_index("Type")
    musical_input.index = musical_signatures.index

    musical_guess, _ = musical.refit.refit(musical_input, musical_signatures, method='likelihood_bidirectional', thresh=0.001)
    musical_guess = torch.from_numpy(musical_guess.transpose().values).to(torch.float)
    return musical_guess

def run_musical_by_mut(data, labels, signatures):
    num_muts = [25,50,100,250,500,1000,5000,10000,50000,100000]
    for i in range(10):
        print(i)
        inputs = data.loc[labels[72]==num_muts[i],]

        start_time = time.time()
        guess_i = get_musical_guess(inputs, signatures)
        end_time = time.time()
        time_taken = end_time - start_time
        print(time_taken)

# Load data
# data_path = DATA + "/datasets/"
# inputs = pd.read_csv(data_path + "test_input.csv", header=None, index_col=None)
# labels = pd.read_csv(data_path + "test_label.csv", header=None, index_col=None)

# GTEx
inputs = pd.read_csv("../../../data/case_study_GTeX/input_data_abund/all_tissues_input_abund_normalized.csv", header=0, index_col=0)

# inputs, labels = subsample(1000, inputs, labels)

# num_mut = labels[72]
print("data loaded")

signatures = pd.read_excel(DATA + '/data.xlsx')

musical_guess = get_musical_guess(inputs, signatures)
tensor_to_csv(musical_guess, 'MuSiCal_guess_GTEx.csv')

# run_musical_by_mut(inputs, labels, signatures)