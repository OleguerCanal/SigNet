import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utilities.plotting import plot_metric_vs_mutations_classifier
from utilities.io import csv_to_tensor, read_model

# MIXED DATASET
num_mut = csv_to_tensor("../../../data/exp_oversample/classifier/test_num_mut.csv", device='cpu')
inputs = csv_to_tensor("../../../data/exp_oversample/classifier/test_input.csv", device='cpu')
label = csv_to_tensor("../../../data/exp_oversample/classifier/test_label.csv", device='cpu').to(torch.int64)

classifier = read_model("../../../trained_models/exp_classifier/classifier")
classifier_guess = classifier(inputs, num_mut)
print(classifier_guess)
classification_cutoff = 0.5
classification_results = (classifier_guess >= classification_cutoff).to(torch.int64)
print(classification_results)
print(label)
plot_metric_vs_mutations_classifier(classification_results, label, num_mut)


# ROC AUC:
# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score

# fpr, tpr, thresh = roc_curve(label.detach().numpy(), classifier_guess.detach().numpy(), pos_label=1)
# print(thresh)
# auc_score = roc_auc_score(label.detach().numpy(), classifier_guess.detach().numpy())
# print(auc_score)

# import matplotlib.pyplot as plt

# # plot roc curves
# plt.plot(fpr, tpr, linestyle='--',color='orange')
# plt.title('ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive rate')
# plt.show()