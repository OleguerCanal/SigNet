import os
import sys

import pandas as pd
import torch
import numpy as np

from utilities.plotting import plot_metric_vs_mutations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_data, read_data_random_yapsa, read_methods_random_data, read_methods_realistic_data
from utilities.metrics import get_classification_metrics
from models.yapsa_inspired_baseline import YapsaInspiredBaseline
from models.finetuner import FineTuner

dev = torch.device("cpu")
_, _, _, _, val_guess_0, val_label = read_data_random_yapsa(dev)

label_random, decompTumor2Sig_guess_random, deconstructSigs_guess_random, MutationalPatterns_guess_random, mutSignatures_guess_random, SignatureEstimationQP_guess_random, YAPSA_guess_random = read_methods_random_data(dev)
label_realistic, decompTumor2Sig_guess_realistic, deconstructSigs_guess_realistic, MutationalPatterns_guess_realistic, mutSignatures_guess_realistic, SignatureEstimationQP_guess_realistic, YAPSA_guess_realistic = read_methods_realistic_data(dev)

list_of_methods = ["decompTumor2Sig", "MutationalPatterns", "mutSignatures", "SignatureEstimationQP","YAPSA"] # "deconstructSigs"

for method in list_of_methods:
    print(method)
    metrics = get_classification_metrics(label_batch=label_random[..., :-1], prediction_batch=globals()[method + "_guess_random"])
    for metric in metrics:
        print(metric, np.round(metrics[metric].detach().numpy(), 8))


# Our methods (baseline with YAPSA and finetuner)
num_classes = 72
signatures_data = pd.read_excel("../data/data.xlsx")
signatures = [torch.tensor(signatures_data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]
input_batch = torch.tensor(pd.read_csv(
        "../data/realistic_test/test_realistic_input.csv", header=None).values, dtype=torch.float)

sf = YapsaInspiredBaseline(signatures)
baseline_guess = sf.get_weights_batch(input_batch)

finetuner_model_name = "finetuner_model_yapsa_random"
finetuner_params = {"num_hidden_layers": 1,
                        "num_units": 1300,
                        "num_classes": 72}

finetuner = FineTuner(**finetuner_params)
finetuner.load_state_dict(torch.load(os.path.join(
            "../trained_models", finetuner_model_name), map_location=torch.device('cpu')))
finetuner.eval()
finetuner_guess = finetuner(mutation_dist=input_batch, weights=baseline_guess)

print(finetuner_guess)

plot_metric_vs_mutations("MAE_p", list_of_methods, baseline_guess, finetuner_guess)
plot_metric_vs_mutations("MAE_n", list_of_methods, baseline_guess, finetuner_guess)
plot_metric_vs_mutations("fp", list_of_methods, baseline_guess, finetuner_guess)
plot_metric_vs_mutations("fn", list_of_methods, baseline_guess, finetuner_guess)


our_methods = ["baseline", "finetuner"]
for method in our_methods:
    print(method)
    metrics = get_classification_metrics(label_batch=label_random[..., :-1], prediction_batch=globals()[method + "_guess"])
    for metric in metrics:
        print(metric, np.round(metrics[metric].detach().numpy(), 8))
