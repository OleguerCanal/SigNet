import os
import sys

from argparse import ArgumentParser
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.classifier_trainer import train_classifier
from utilities.io import update_dict, read_config

DEFAULT_CONFIG_FILE = ["configs/classifier_bayesian.yaml"]

if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--config_file', action='store', nargs=1, type=str, required=False, default=DEFAULT_CONFIG_FILE,
        help=f'Path to yaml file containing all needed parameters.\nThey can be overwritten by command-line arguments'
    )

    parser.add_argument(
        '--model_id', action='store', nargs=1, type=str, required=False,
        help=f'Unique id given to the trained model.'
    )

    # Data args
    parser.add_argument(
        '--sigmoid_params', action='store', nargs=1, type=float, required=False,
        help='Sigmoid parameters for normalization.'
    )

    # Train args
    parser.add_argument(
        '--batch_size', action='store', nargs=1, type=int, required=False,
        help='Training batch size.'
    )
    parser.add_argument(
        '--lr', action='store', nargs=1, type=float, required=False,
        help='Learning rate.'
    )
    
    # Model args
    parser.add_argument(
        '--num_neurons', action='store', nargs=1, type=int, required=False,
        help='Neurons of the positive error network.'
    )
    parser.add_argument(
        '--num_hidden_layers', action='store', nargs=1, type=int, required=False,
        help='Layers of the positive error network.'
    )

    _args = parser.parse_args()

    # Read & config
    config = read_config(path=getattr(_args, "config_file")[0])
    config = update_dict(config=config, args=_args)
    
    print("Using config:", config)
    score = train_classifier(config=config)
    fout = open("../tmp/classifier_score_%s.txt"%config["model_id"], 'w')
    fout.write(str(score))
    fout.close()
