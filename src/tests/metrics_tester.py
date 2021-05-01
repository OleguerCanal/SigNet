import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.metrics import *
from baseline import SignatureFinder
from model_tester import *

metrics = {
    "mse" : get_MSE,
    "cos" : get_negative_cosine_similarity,
    "cross_ent" : get_cross_entropy2,
    "KL" : get_kl_divergence,
    "JS" : get_jensen_shannon,
    "W" : get_wasserstein_distance,
}

if __name__ == "__main__":

    num_classes = 72

    label_batch = torch.tensor(pd.read_csv("../data/validation_label_w01.csv", header=None).values, dtype=torch.float)
    guessed_labels = torch.tensor(pd.read_csv("../data/validation_w01_baseline_mse.csv", header=None).values, dtype=torch.float)
    #guessed_labels = torch.tensor(pd.read_csv("../data/deconstructSigs_validation_w01.csv", header=None).values, dtype=torch.float)

    # # Get metrics
    model_tester = ModelTester(num_classes=num_classes)
    model_tester.test(guessed_labels=guessed_labels, true_labels=label_batch[:,:72])

    # False negatives:
    label_sigs_list, predicted_sigs_list = model_tester.probs_batch_to_sigs(label_batch[:,:72], guessed_labels, 0.05, num_classes)
    FN = sum(predicted_sigs_list == num_classes)
    print('Number of FN:', FN)

    # False positives:
    FP = sum(label_sigs_list == num_classes)
    print('Number of FP:', FP)