import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import Classifier
from utilities.io import csv_to_tensor

def accuracy(prediction, label):
    threshold = 0.5
    prediction = (prediction>threshold).float()*1
    return torch.sum(prediction == label)/torch.numel(prediction)*100

def false_realistic(prediction, label):
    threshold = 0.5
    prediction = (prediction>threshold).float()*1
    return torch.sum(label[prediction == 1] == 0)/torch.numel(label[prediction == 1])*100

def false_random(prediction, label):
    threshold = 0.5
    prediction = (prediction>threshold).float()*1
    return torch.sum(label[prediction == 0] == 1)/torch.numel(label[prediction == 0])*100

def plot_metric_vs_mutations(guess, label, num_muts_list, plot_path = '../../plots/plot_classifier.png'):
    fig, axs = plt.subplots(3)
    fig.suptitle("Metrics vs Number of Mutations")
    
    num_muts_list = num_muts_list[num_muts_list<=100000]
    num_muts = np.unique(num_muts_list.detach().numpy())
    
    values = np.zeros((3, len(num_muts)))
    for i in range(len(num_muts)):
        values[0,i] = accuracy(label=label[2000*i:2000*(i+1),:], prediction=guess[2000*i:2000*(i+1),:])
        values[1,i] = false_realistic(label=label[2000*i:2000*(i+1),:], prediction=guess[2000*i:2000*(i+1),:])
        values[2,i] = false_random(label=label[2000*i:2000*(i+1),:], prediction=guess[2000*i:2000*(i+1),:])
        
    axs[0].plot(np.log10(num_muts), values[0,:])
    axs[0].set_ylabel("Accuracy (%)")

    axs[1].plot(np.log10(num_muts), values[1,:])
    axs[1].set_ylabel("False Realistic (%)")

    axs[2].plot(np.log10(num_muts), values[2,:])
    axs[2].set_ylabel("False Random (%)")

    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
    #fig.savefig(plot_path)

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