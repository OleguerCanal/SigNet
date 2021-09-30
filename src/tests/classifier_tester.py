import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import Classifier
from utilities.plotting import plot_metric_vs_mutations
from utilities.io import csv_to_tensor


experiment_id = "exp_classifier"

classifier_model_name = "classifier_1"
classifier_params = {"num_hidden_layers": 1,
                    "num_units": 300}

input = csv_to_tensor("../../data/" + experiment_id + "/test_input.csv", device='cpu')
num_mut = csv_to_tensor("../../data/" + experiment_id + "/test_num_mut.csv", device='cpu')
label = csv_to_tensor("../../data/" + experiment_id + "/test_label.csv", device='cpu')

classifier = Classifier(**classifier_params)
classifier.load_state_dict(torch.load(os.path.join(
    "../../trained_models", experiment_id, classifier_model_name), map_location=torch.device('cpu')))
classifier.eval()  

classifier_guess = classifier(input, num_mut)

plot_metric_vs_mutations(classifier_guess, label, num_mut)