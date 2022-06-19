import os
import sys

#import numpy as np
from argparse import ArgumentParser
import pandas as pd
import torch

from signet import DATA
from signet.modules.signet_module import SigNet
from signet.utilities.io import csv_to_tensor, write_final_outputs

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_data', action='store', nargs=1, type=str, required=False, default=DATA + "datasets/real_input.csv",
        help=f'Path to the input data to be analyzed. By default it will use PCAWG dataset'
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
        '--figures', action='store', nargs=1, type=bool, required=False, default=[True],
        help=f'Boolean. Whether to compute plots for the output. Default: "False".'
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Load signet
    signet = SigNet(opportunities_name_or_path=args.normalization)

    # Read and prepare data
    inputs = csv_to_tensor(file=args.input_data,
                           header=0,
                           index_col=0)
    
    # Run signet
    results = signet(mutation_vec=mutation_data)

    # Write final outputs
    write_final_outputs(weights=results["finetuner_guess"],
                        lower_bound=results["error_lower"], 
                        upper_bound=results["error_upper"],
                        classification=results["classification"],
                        reconstruction_error=0,
                        input_file=args.input_data,
                        output_path=args.output,
                        name="")

    # Plot figures
    if plot_figs:
        sig_names = list(pd.read_excel(DATA + "data.xlsx").columns)[1:]
        for i in range(weight_guess.shape[0]):
            plot_weights(weight_guess[i,:], upper_bound[i,:], lower_bound[i,:], sig_names, output_path + "/plots/plot_sample_%s.png"%str(i))
        # plot_reconstruction(normalized_input, signet.baseline_guess, signet.signatures, list(range(weight_guess.shape[0])), output_path + "/plots/baseline_reconstruction")