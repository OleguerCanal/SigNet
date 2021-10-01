import os
import sys

import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.finetuner_trainer import FinetunerTrainer
from utilities.io import read_data

config = {
    # IDs
    "experiment_id": "exp_0",
    "model_id": "test_new_saving",

    # Training params
    "source": "random",
    "iterations": 10,
    "num_classes": 72,
    "fp_param": 0.001,
    "fn_param": 0.001,
    "sigmoid_params": [500,150],      # Low num muts: [500, 150]. Large num muts: [50000, 10000]
    "batch_size": 500,
    "lr": 0.0001,

    # Network params
    "num_hidden_layers": 2,
    "num_neurons": 600,
}

models_path = os.path.join("../trained_models", config["experiment_id"])
finetuner_path = os.path.join(models_path,
                    "finetuner_" + config["source"] + "_" + config["model_id"])

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = "cpu"
    device = torch.device(dev)
    print("Using device:", dev)

    wandb.init(project='finetuner_two_trains',
               entity='sig-net',
               config=config,
               name=config["model_id"])

    train_data, val_data = read_data(device=dev,
                                     experiment_id=config["experiment_id"],
                                     source=config["source"])

    trainer = FinetunerTrainer(iterations=config["iterations"],  # Passes through all dataset
                               train_data=train_data,
                               val_data=val_data,
                               num_classes=config["num_classes"],
                               fp_param=config["fp_param"],
                               fn_param=config["fn_param"],
                               sigmoid_params=config["sigmoid_params"],
                               device=device,
                               model_path=finetuner_path)

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_hidden_layers=config["num_hidden_layers"],
                                num_units=config["num_neurons"],
                                plot=True)
