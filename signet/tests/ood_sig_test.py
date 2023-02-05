import os
import sys

import numpy as np
import pandas as pd
import torch

from signet import DATA
from signet.modules.signet_module import SigNet
from signet.utilities.data_generator import DataGenerator
from signet.utilities.io import read_signatures, csv_to_tensor


def generate_and_guess(labels):
    signatures = read_signatures(file=os.path.join(DATA, "data.xlsx"),
                                 mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx"))

    data_generator = DataGenerator(signatures=signatures, normalize=False)
    inputs, labels = data_generator.make_input(labels=labels, split="train", large_low="large")

    # Guess
    signet = SigNet()
    inputs_df = pd.DataFrame(inputs.numpy(), index=list(pd.read_excel(os.path.join(DATA, "data.xlsx")).columns)[1:])
    results = signet(mutation_dataset=inputs_df, nworkers=-1, numpy=False)

    return results.weights, results.classification


if __name__ == "__main__":
    real_data = csv_to_tensor(DATA + "/real_data/sigprofiler_not_norm_PCAWG.csv", header=0, index_col=0)
    labels = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)

    # SUbset for debugging TODO: remove
    labels = labels[:10, :]

    """torch.sum(labels, dim=0)):  (this is to see which ones are zero)
       [3.1198e+03, 6.4312e+02, 1.6833e+03, 5.0126e+02, 1.0887e+04, 3.4905e+01,
        5.6165e+02, 1.6601e+02, 2.6946e+01, 3.0822e+01, 1.0344e+02, 2.7881e+02,
        4.0360e+01, 1.5465e+01, 1.0614e+01, 4.2280e+02, 6.1960e+02, 6.7205e+00,
        1.6920e+01, 4.3577e+01, 1.7241e+02, 3.6390e+02, 8.6510e+02, 2.5683e+01,
        5.0930e+00, 5.6150e+00, 6.3954e+01, 9.9882e+00, 7.3592e+00, 0.0000e+00,
        4.0384e+01, 0.0000e+00, 3.3613e+01, 1.6921e+02, 5.2818e+01, 1.2013e+01,
        5.6520e+00, 4.8003e+00, 1.1589e+01, 2.3450e+01, 7.8677e+01, 6.0066e+01,
        1.8791e+01, 4.8162e+01, 6.2619e+03, 7.2469e+01, 0.0000e+00, 2.1053e+00,
        9.4051e+01, 4.2647e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.8211e+01, 6.5693e+00, 3.3525e+00, 1.1205e+00, 0.0000e+00,
        1.2480e+01, 0.0000e+00, 1.9151e+01, 0.0000e+00, 1.2872e+01, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        4.6405e+08]
    """

    signature_to_test = -2
    set_weight, guessed_weight, unfamiliar_prop = [], [], []
    for fake_weight in np.linspace(0, 1, 10):
        labels[:, signature_to_test] = fake_weight
        # normalize:
        labels = labels / torch.sum(labels, dim=1, keepdim=True)

        print(labels.shape)
        guesses, labels = generate_and_guess(labels)
        
        guessed_weight, unfamiliar_prop = torch.mean(guesses[:, signature_to_test])

        set_weight.append(fake_weight)
        guessed_weight.append(guessed_weight)
        unfamiliar_prop.append(unfamiliar_prop)

    # TODO: Plot this
    print(set_weight)
    print(guessed_weight)
    print(unfamiliar_prop)