from argparse import ArgumentParser
import os
import sys

import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.error_trainer import train_errorfinder
from utilities.io import update_dict

# Default config
config = {
    # IDs & paths
    "data_id": "exp_0",
    "model_id": "exp_errorfiner_loss/test_0",
    "finetuner_id": "exp_2_nets/finetuner_realistic",
    "models_dir": "../trained_models",

    # Training params
    "source": "mixed",
    "iterations": 1,
    "num_classes": 72,
    "batch_size": 500,
    "lr": 0.0001,
    "loss_params": {
        "lagrange_missclassification": 7e-3,
        "lagrange_pnorm": 1e4,
        "lagrange_smalltozero": 1.0,
        "pnorm_order": 5.0,
    },

    # WANDB params
    "enable_logging": True,
    "wandb_project_id": "errorfinder",

    # Network params
    "num_hidden_layers_pos": 2,
    "num_neurons_pos": 1000,
    "num_hidden_layers_neg": 2,
    "num_neurons_neg": 1000,

    # Misc
    "device": "cuda",
}

if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--model_id', action='store', nargs=1, type=str, required=False,
        help=f'Unique id given to the trained model.'
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
        '--num_neurons_pos', action='store', nargs=1, type=int, required=False,
        help='Neurons of the positive error network.'
    )
    parser.add_argument(
        '--num_hidden_layers_pos', action='store', nargs=1, type=int, required=False,
        help='Layers of the positive error network.'
    )
    parser.add_argument(
        '--num_neurons_neg', action='store', nargs=1, type=int, required=False,
        help='Neurons of the negative error network.'
    )
    parser.add_argument(
        '--num_hidden_layers_neg', action='store', nargs=1, type=int, required=False,
        help='Layers of the negative error network.'
    )

    # Loss args
    parser.add_argument(
        '--lagrange_missclassification', action='store', nargs=1, type=float, required=False,
        help='Weight of misclassification in the loss.'
    )
    parser.add_argument(
        '--lagrange_pnorm', action='store', nargs=1, type=float, required=False,
        help='Weight of big errors in the loss.'
    )
    parser.add_argument(
        '--lagrange_smalltozero', action='store', nargs=1, type=float, required=False,
        help='Weight of not sending small values to zero in the loss.'
    )
    parser.add_argument(
        '--pnorm_order', action='store', nargs=1, type=int, required=False,
        help='Norm used in the loss.'
    )
    _args = parser.parse_args()

    # Update config 
    config = update_dict(config=config, args=_args)
    config["loss_params"] = update_dict(config=config["loss_params"], args=_args)

    print(config)
    score = train_errorfinder(config=config)
    fout = open("../tmp/score_%s.txt"%config["model_id"])
    fout.write(str(score))
    fout.close()