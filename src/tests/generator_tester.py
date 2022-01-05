import os
import sys

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline import Baseline
from utilities.io import read_data_generator, read_signatures, read_test_data, read_model
from utilities.plotting import plot_all_metrics_vs_mutations, plot_metric_vs_mutations, plot_metric_vs_sigs

model_directory = "../../trained_models/exp_good/generator"

# Load data
train_data, val_data = read_data_generator(device="cpu", data_folder="../../data")

# Load generator and get predictions
generator = read_model(model_directory)
generator_output, mean, var = generator(x=val_data.inputs, noise=False)


def plot_weights_comparison(true_labels, guessed_labels, sigs_names):
    num_classes = len(guessed_labels)
    fig, ax = plt.subplots()
    ax.bar(range(num_classes),guessed_labels, align='center', width=0.2, alpha=0.5, ecolor='black', capsize=10)
    ax.bar(np.array(range(num_classes))+0.2, true_labels, width=0.2, align='center')
    ax.set_ylim([0,1])
    ax.set_ylabel('Weights')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels(sigs_names, rotation='vertical')
    ax.set_title('Signature decomposition')
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()

input0 = val_data.inputs[0,:]
output0, mean, var = generator(x=input0, noise=False)
data = pd.read_excel("../../data/data.xlsx")
plot_weights_comparison(input0.detach().numpy(), output0.detach().numpy(), list(data.columns)[1:])
print(val_data.inputs)
print(generator_output)
print(mean)
print(var) 