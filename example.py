import os
import sys

#import numpy as np
from argparse import ArgumentParser
import pandas as pd
import torch

from signet.modules.signet_module import SigNet
from signet import DATA as SIGNET_DATA

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_data', action='store', nargs=1, type=str, required=True,
        help=f'Path to the input data to be analyzed.'
    )

    parser.add_argument(
        '--normalization', action='store', nargs=1, type=str, required=False, default=[None],
        help=f'The kind of normalization to be applied to the data. Should be either "None" (default), "exome", "genome" or a path to a file with the oppportunities.'
    )
    
    parser.add_argument(
        '--output', action='store', nargs=1, type=str, required=False, default=["Output"],
        help=f'Path to folder where all the output files will be saved. Default: "Output".'
    )

    parser.add_argument(
        '--figures', action='store', nargs=1, type=bool, required=False, default=[False],
        help=f'Boolean. Whether to compute plots for the output. Default: "False".'
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse command-line arguments
    config = parse_args()

    input_file_path = "data/datasets/real_input.csv"
    output_path = "data/case_study_Hypoxia/new_SigNet/" 
    opportunities = None
    plot_figs = False

    signet = SigNet(opportunities_name_or_path=opportunities, signatures_path="../../data/data.xlsx")

    input_file = pd.read_csv(input_file_path, header=0, index_col=0, sep=',')
    mutation_data = torch.tensor(input_file.values, dtype=torch.float)
    print(mutation_data)
    weight_guess, upper_bound, lower_bound, classification, normalized_input = signet(mutation_vec=mutation_data)

    # Write final outputs
    write_final_outputs(weight_guess, lower_bound, upper_bound, classification, 0, input_file, output_path)


    # Plot figures
    if plot_figs:
        sig_names = list(pd.read_excel("../../data/data.xlsx").columns)[1:]
        for i in range(weight_guess.shape[0]):
            plot_weights(weight_guess[i,:], upper_bound[i,:], lower_bound[i,:], sig_names, output_path + "/plots/plot_sample_%s.png"%str(i))
        # plot_reconstruction(normalized_input, signet.baseline_guess, signet.signatures, list(range(weight_guess.shape[0])), output_path + "/plots/baseline_reconstruction")