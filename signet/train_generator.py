import os
import sys

from argparse import ArgumentParser
import torch
import pandas as pd
import wandb

from signet.trainers import train_generator
from signet.utilities.io import update_dict, read_config, write_result

DEFAULT_CONFIG_FILE = ["configs/generator/generator.yaml"]

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
    parser.add_argument(
        '--type', action='store', nargs=1, type=str, required=False,
        help=f'Type of the data that we are going to use.'
    )
    # Train args
    parser.add_argument(
        '--lagrange_param', action='store', nargs=1, type=float, required=False,
        help='Lagrange Parameter'
    )
    parser.add_argument(
        '--batch_size', action='store', nargs=1, type=int, required=False,
        help='Training batch size.'
    )
    parser.add_argument(
        '--lr_encoder', action='store', nargs=1, type=float, required=False,
        help='Learning rate of the encoder.'
    )
    parser.add_argument(
        '--lr_decoder', action='store', nargs=1, type=float, required=False,
        help='Learning rate of the decoder.'
    )
    # Model args
    parser.add_argument(
        '--latent_dim', action='store', nargs=1, type=int, required=False,
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
    val_mse, val_KL = train_generator(config=config)
    model_results = pd.DataFrame({"batch_size": [config["batch_size"]],
                                  "lr_encoder": [config["lr_encoder"]],
                                  "lr_decoder": [config["lr_decoder"]],
                                  "num_hidden_layers": [config["num_hidden_layers"]],
                                  "latent_dim": [config["latent_dim"]],
                                  "lagrange_param": [config["lagrange_param"]],
                                  "adapted_lagrange_param": [""],
                                  "batch_size_factor": [""],
                                  "val_mse": [val_mse],
                                  "val_KL": [val_KL],
                                  "val_loss": [""]})
    model_results.to_csv("../tmp/generator_models.csv",
                         header=False, index=False, mode="a")
