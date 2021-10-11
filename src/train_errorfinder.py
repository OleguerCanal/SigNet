import os
import sys

import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.finetuner import baseline_guess_to_finetuner_guess
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data

config = {
    # IDs
    "experiment_id": "exp_0",
    "model_id": "test_0",
    "finetuner_id": "",

    # Training params
    "source": "mixed",
    "iterations": 40,
    "num_classes": 72,
    "batch_size": 500,
    "lr": 0.0001,

    # Network params
    "num_hidden_layers_pos": 2,
    "num_neurons_pos": 1000,
    "num_hidden_layers_neg": 2,
    "num_neurons_neg": 1000,
}

# Paths
models_path = os.path.join("../trained_models", config["experiment_id"])
errorfinder_path = os.path.join(models_path, config["model_id"])
finetuner_path = os.path.join(models_path, config["finetuner_id"])

if __name__ == "__main__":
    # Select training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    wandb.init(project='errorfinder_loss',
               entity='sig-net',
               config=config)

    # Load data
    train_data, val_data = read_data(experiment_id=config["experiment_id"],
                                     source=config["source"],
                                     device="cpu")

    train_data = baseline_guess_to_finetuner_guess(trained_finetuner_dir=finetuner_path,
                                                   data=train_data)
    val_data = baseline_guess_to_finetuner_guess(trained_finetuner_dir=finetuner_path,
                                                 data=val_data)

    train_data.to(device=device)
    val_data.to(device=device)
    torch.cuda.empty_cache()

    trainer = ErrorTrainer(iterations=config["iterations"],  # Passes through all dataset
                           train_data=train_data,
                           val_data=val_data,
                           num_classes=config["num_classes"],
                           device=device,
                           model_path=errorfinder_path)

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_neurons_pos=config["num_neurons_pos"],
                                num_neurons_neg=config["num_neurons_neg"],
                                num_hidden_layers_pos=config["num_hidden_layers_pos"],
                                num_hidden_layers_neg=config["num_hidden_layers_neg"],
                                plot=True)
