import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import copy

from signet import DATA
from signet.modules.signet_module import SigNet
from signet.utilities.data_generator import DataGenerator
from signet.utilities.io import read_signatures, csv_to_tensor


def generate_and_guess(labels):
    signatures = read_signatures(file=os.path.join(DATA, "data.xlsx"),
                                 mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx"))

    data_generator = DataGenerator(signatures=signatures, normalize=False)
    
    all_num_muts = True
    if all_num_muts == False:
        # For a given number of mutations:
        muts = [10000]*10000
        inputs, labels = data_generator.make_input(labels=labels, split="train", large_low="low", nummuts=muts)
    else:
        # For the training set:
        inputs, labels = data_generator.make_input(labels=labels, split="train", large_low="low", nummuts=None)

    # Guess
    signet = SigNet()
    results = signet(mutation_dataset=inputs, nworkers=8, numpy=False)

    return results.weights, results.classification


if __name__ == "__main__":
    col_names = list(pd.read_excel(os.path.join(DATA, "data.xlsx")).columns)[1:]
    # inputs = torch.rand(10, 96)
    # inputs_df = pd.DataFrame(inputs.numpy(), columns=col_names)


    real_data = csv_to_tensor(DATA + "/real_data/sigprofiler_not_norm_PCAWG.csv", header=0, index_col=0)
    labels = torch.cat([real_data, torch.zeros(real_data.size(0), 7).to(real_data)], dim=1)

    # Subset for debugging TODO: remove
    # labels = labels[:10, :]

    """torch.sum(labels, dim=0)):  (this is to see which ones are zero)
        3.1198e+02, 6.4312e+01, 1.6833e+02, 5.0126e+01, 1.0887e+03, 3.4905e+00,
        5.6165e+01, 1.6601e+01, 2.6946e+00, 3.0822e+00, 1.0344e+01, 2.7881e+01,
        4.0360e+00, 1.5465e+00, 1.0614e+00, 4.2280e+01, 6.1960e+01, 6.7205e-01,
        1.6920e+00, 4.3577e+00, 1.7241e+01, 3.6390e+01, 8.6510e+01, 2.5683e+00,
        5.0930e-01, 5.6150e-01, 6.3954e+00, 9.9882e-01, 7.3592e-01, 0.0000e+00,
        4.0384e+00, 0.0000e+00, 3.3613e+00, 1.6921e+01, 5.2818e+00, 1.2013e+00,
        5.6520e-01, 4.8003e-01, 1.1589e+00, 2.3450e+00, 7.8677e+00, 6.0066e+00,
        1.8791e+00, 4.8162e+00, 6.2619e+02, 7.2469e+00, 0.0000e+00, 2.1053e-01,
        9.4051e+00, 4.2647e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 1.8211e+00, 6.5693e-01, 3.3525e-01, 1.1205e-01, 0.0000e+00,
        1.2480e+00, 0.0000e+00, 1.9151e+00, 0.0000e+00, 1.2872e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])
    """

    list_signatures_to_test = [29, 31, 46, 50, 51, 52, 53, 54, 59,61, 63, 65, 66, 67, 68, 69, 70, 71]
    fig,ax = plt.subplots(6,3, figsize=(80, 60))
    for i, signature_to_test in enumerate(list_signatures_to_test):
        set_weights, guessed_weights, unfamiliar_props = [], [], []
        for fake_weight in np.linspace(0, 1, 11):
            # print("fake_weight", fake_weight)
            labels_new = copy.deepcopy(labels)
            labels_new[:, signature_to_test] = fake_weight
            # normalize:
            labels_new = labels_new / torch.sum(labels_new, dim=1, keepdim=True)

            guesses, classification = generate_and_guess(labels_new)
            
            guessed_weight = torch.mean(guesses[:, signature_to_test])
            # if guessed_weight>0:
            #     print('Weight>0!!!!!')
            #     print(guessed_weight)
            unfamiliar_prop = torch.mean((classification < 0.5).to(torch.float))

            set_weights.append(fake_weight)
            guessed_weights.append(guessed_weight)
            unfamiliar_props.append(unfamiliar_prop)

        # TODO: Plot this
        # print(set_weights)
        # print(guessed_weights)
        # print(unfamiliar_props)

        c = i%3
        r = i//3
        ax[r,c].plot(set_weights, guessed_weights, '-o', label="guessed weights")
        ax[r,c].plot(set_weights, unfamiliar_props, '-o', label="unfamiliar proportion" )
        ax[r,c].plot([0,1], [0,1], '-r')
        ax[r,c].set_xlabel('Weight of signature %s'%col_names[signature_to_test])
        ax[r,c].set_title(col_names[signature_to_test])
        ax[r,c].set_ylim([0,1])
    # fig.tight_layout()
    plt.legend()
    plt.show()