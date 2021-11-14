import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utilities.plotting import plot_metric_vs_mutations_classifier
from utilities.io import csv_to_tensor, read_model

# MIXED DATASET
# inputs = csv_to_tensor("../../../data/exp_classifier/test_input.csv", device='cpu')
# num_mut = csv_to_tensor("../../../data/exp_classifier/test_num_mut.csv", device='cpu')
# label = csv_to_tensor("../../../data/exp_classifier/test_label.csv", device='cpu')

# ALL REALISTIC
inputs = csv_to_tensor("../../../data/exp_final/test_realistic/test_realistic_input.csv", device='cpu')
num_mut = csv_to_tensor("../../../data/exp_final/test_realistic/test_realistic_label.csv", device='cpu')[:, -1].unsqueeze(1)
label = torch.ones(num_mut.shape)

classifier = read_model("../../../trained_models/exp_final_2/classifier_2")
classifier_guess = classifier(inputs, num_mut)

plot_metric_vs_mutations_classifier(classifier_guess, label, num_mut)