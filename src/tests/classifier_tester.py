import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import Classifier
from utilities.plotting import plot_metric_vs_mutations_classifier
from utilities.io import csv_to_tensor, read_model


experiment_id = "exp_classifier"

classifier_model_name = "classifier"

input = csv_to_tensor("../../data/" + experiment_id + "/test_input.csv", device='cpu')
num_mut = csv_to_tensor("../../data/" + experiment_id + "/test_num_mut.csv", device='cpu')
label = csv_to_tensor("../../data/" + experiment_id + "/test_label.csv", device='cpu')

classifier = read_model("../../trained_models/%s/%s"%(experiment_id,classifier_model_name))

classifier_guess = classifier(input, num_mut)

plot_metric_vs_mutations_classifier(classifier_guess, label, num_mut)