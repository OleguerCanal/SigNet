import os
import sys

import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.classifier_trainer import ClassifierTrainer
from utilities.io import read_data, read_data_classifier

config = {
    # IDs
    "experiment_id": "exp_classifier",
    "model_id": "1",

    # Training params
    "iterations": 20,
    "sigmoid_params": [5000,1000],
    "batch_size": 500,
    "lr": 0.0001,

    # Network params
    "num_hidden_layers": 2,
    "num_neurons": 300,
}

models_path = os.path.join("../trained_models", config["experiment_id"])
classifier_path = os.path.join(models_path,
                    "classifier_" + config["model_id"])

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = "cpu"
    device = torch.device(dev)
    print("Using device:", dev)

    wandb.init(project='classifier',
               entity='sig-net',
               config=config,
               name=config["model_id"])

    train_data, val_data = read_data_classifier(device=dev,
                                     experiment_id=config["experiment_id"])

    trainer = ClassifierTrainer(iterations=config["iterations"],  # Passes through all dataset
                               train_data=train_data,
                               val_data=val_data,
                               sigmoid_params=config["sigmoid_params"],
                               device=device,
                               model_path=classifier_path)

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_hidden_layers=config["num_hidden_layers"],
                                num_units=config["num_neurons"],
                                plot=True)
