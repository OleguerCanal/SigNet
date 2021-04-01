import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.metrics import *
from baseline import SignatureFinder

metrics = {
    "mse" : get_MSE,
    "cos:" : get_cosine_similarity,
    "cross_ent" : get_cross_entropy2,
    "KL" : get_kl_divergence,
    "JS" : get_jensen_shannon,
    "W" : get_wasserstein_distance,
}

if __name__=="__main__":
    # true_label = torch.tensor([0, 0, 0.3, 0, 0.2, 0.5, 0, 0], dtype=torch.double).unsqueeze(0)
    # prediction_1 = torch.tensor([0, 0, 0.4, 0, 0.15, 0.45, 0, 0], dtype=torch.double).unsqueeze(0)
    # prediction_2 = torch.tensor([0, 0.1, 0.3, 0, 0.15, 0.45, 0, 0], dtype=torch.double).unsqueeze(0)

    # for metric_name in list(metrics.keys()):
    #     a = metrics[metric_name](prediction_1, true_label).item()
    #     b = metrics[metric_name](prediction_2, true_label).item()
    #     print(metric_name, a, b)

    num_classes = 5
    data = pd.read_excel("../data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]
    sf = SignatureFinder(signatures)
    label = np.array([0.5, 0.3, 0.1, 0.1, 0.0])
    signature = label[0]*sf.signatures[:, 0] + label[1]*sf.signatures[:, 1] +\
        label[2]*sf.signatures[:, 2] + label[3]*sf.signatures[:, 3]
    
    for optimizer_metric in list(metrics.keys()):
        sf.metric = metrics[optimizer_metric]
        print("Optimizing", optimizer_metric)
        pred = sf.get_weights(signature)
        pred = torch.tensor(pred, dtype=torch.double).unsqueeze(0)
        true = torch.tensor(label, dtype=torch.double).unsqueeze(0)
        for wight_metric in list(metrics.keys()):
            val = metrics[wight_metric](pred, true).item()
            print(wight_metric, ":", np.round(val, decimals=4), ". sum:", np.round(torch.sum(pred).item(), decimals=4))
