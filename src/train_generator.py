import os
import sys

from argparse import ArgumentParser
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.generator_trainer import train_generator
from utilities.io import update_dict, read_config, write_result

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
    score = train_generator(config=config)
    write_result(score, "../tmp/classifier_score_%s.txt"%config["model_id"])
