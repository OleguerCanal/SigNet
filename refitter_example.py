import logging

#import numpy as np
from argparse import ArgumentParser
import pandas as pd

from signet import DATA
from signet.modules.signet_module import SigNet

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment_id', action='store', nargs=1, type=str, required=False, default="test_0",
        help=f"Name of this inference's results"
    )

    parser.add_argument(
        '--input_data', action='store', nargs=1, type=str, required=False, default=DATA + "/datasets/example_input.csv",
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
        '--plot_figs', action='store', nargs=1, type=bool, required=False, default=[False],
        help=f'Boolean. Whether to compute plots for the output. Default: "False".'
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Read data
    mutations = pd.read_csv(args.input_data, header=0, index_col=0)

    # Load & Run signet
    signet = SigNet(opportunities_name_or_path=args.normalization[0])
    results = signet(mutation_dataset=mutations)
    print("Results obtained!")

    # Extract results
    w, u, l, c, _ = results.convert_output()

    # Store results
    results.save(path='Output', name=args.experiment_id)

    # Plot figures
    results.plot_results(save=True)
