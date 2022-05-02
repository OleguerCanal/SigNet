import os
from socket import TIPC_LOW_IMPORTANCE
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utilities.io import csv_to_tensor, tensor_to_csv
from utilities.data_generator import DataGenerator

experiment_id = "exp_all"

# Creat test set:
real_weights = csv_to_tensor("../../../data/real_data/sigprofiler_not_norm_PCAWG.csv", device='cpu', header=0, index_col=0)
real_weights = real_weights[:, :-1]
test_input, test_label = DataGenerator.make_input(real_weights, "test", "", normalize=True)

tensor_to_csv(test_input, "../../../data/%s/test_input.csv"%experiment_id)
tensor_to_csv(test_label, "../../../data/%s/test_label.csv"%experiment_id)