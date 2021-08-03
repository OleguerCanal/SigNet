import os
import sys

import pandas as pd
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_data, read_data_yapsa
from utilities.metrics import get_classification_metrics

dev = torch.device("cpu")
_, _, _, _, val_guess_0, val_label = read_data_yapsa(dev)

metrics = get_classification_metrics(label_batch=val_label[..., :-1], prediction_batch=val_guess_0)

for metric in metrics:
    print(metric, np.round(metrics[metric].detach().numpy(), 8))