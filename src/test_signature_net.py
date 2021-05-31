import os
import sys

import numpy as np
import pandas as pd
import torch


from models.signature_net import SignatureNet
from tests.model_tester import ModelTester
from utilities.metrics import get_jensen_shannon


def read_data():
    data_folder = "../data/"
    input_path = "realistic_data/ground.truth.syn.catalog_test.csv"
    label_path = "realistic_data/ground.truth.syn.exposures_test.csv"
    train_path = "data.xlsx"

    train_data = pd.read_excel(data_folder + train_path, index_col=[0, 1])
    mutations_ids = train_data.index.tolist()
    signatures_ids = train_data.columns.values.tolist()

    input_df = pd.read_csv(os.path.join(data_folder, input_path), index_col=[0, 1])

    # The test set has less signatures and in different order, we must match them with the training set
    label_df = pd.read_csv(os.path.join(data_folder, label_path), index_col=0)
    label_df_fixed = pd.DataFrame(0, index=signatures_ids, columns=label_df.columns.values.tolist())
    label_df_fixed.update(label_df)

    # print(mutations_ids)
    # print(input_df.index.tolist())
    # print(signatures_ids)
    # print(label_df.index.tolist())
    # print(input_df.index.tolist() == mutations_ids)
    # print(label_df.index.tolist() == signatures_ids)
    # print(label_df_fixed.index.tolist() == signatures_ids)

    # To tensor (transpose since we want batch-first)
    input_tensor = torch.transpose(torch.from_numpy(np.array(input_df.values, dtype=np.float32)), 0, 1).to("cpu")[:1000, ...]
    label_tensor = torch.transpose(torch.from_numpy(np.array(label_df_fixed.values, dtype=np.float32)), 0, 1).to("cpu")[:1000, ...]

    # Normalize
    label_tensor = label_tensor / torch.sum(label_tensor, dim=1).reshape(-1,1)
    return train_data, input_tensor, label_tensor


if __name__=="__main__":
    num_classes = 72

    data, test_input, test_label = read_data()

    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                  for i in range(72)][:num_classes]
    signature_finder_params = {"signatures": signatures,
                               "metric": get_jensen_shannon,
                               "n_workers": 10}

    finetuner_model_name = "finetuner_model_2"  # NOTE! Maybe you have a better one!!
    finetuner_params = {"num_hidden_layers": 1,
                        "num_units": 1500,
                        "num_classes": 72}

    error_finder_model_name = "error_finder_model_new_loss_bayesian_1"  # NOTE! Maybe you have a better one!!
    error_learner_params = {"num_hidden_layers_pos": 3,
                            "num_units_pos": 1500,
                            "num_hidden_layers_neg": 1,
                            "num_units_neg": 700,
                            "normalize_mut": 2e4}

    signature_net = SignatureNet(signature_finder_params, finetuner_params, error_learner_params,
                                finetuner_model_name, error_finder_model_name, path_opportunities=None,
                                models_path="../trained_models/")

    weight0, inferred_tensor, pos, neg = signature_net(mutation_vec=test_input)

    model_tester = ModelTester(num_classes=72)
    model_tester.test(guessed_labels=inferred_tensor, true_labels=test_label)