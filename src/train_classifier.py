import os
import sys

import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.classifier_trainer import train_classifier

config = {
    # IDs & paths
    "data_id": "exp_classifier",
    "model_id": "exp_classifier/classifier_test",
    "models_dir": "../trained_models",

    # Training params
    "iterations": 20,
    "sigmoid_params": [5000,1000],
    "batch_size": 500,
    "lr": 0.0001,

    # Network params
    "num_hidden_layers": 2,
    "num_neurons": 300,

    # WANDB params
    "enable_logging": True,
    "wandb_project_id": "classifier",

    # Misc
    "device": "cuda"
}

if __name__ == "__main__":
    score = train_classifier(config=config)
    print(score)