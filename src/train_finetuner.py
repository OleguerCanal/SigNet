import os
import sys

import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.finetuner_trainer import FinetunerTrainer
from utilities.io import read_data, read_data_type

config = {
    # IDs
    "experiment_id": "exp_two_trainings",
    "model_id": "large_mut_train",

    # Training params
    "source": "mixed",
    "iterations": 40,
    "num_classes": 72,
    "fp_param": 0.001,
    "fn_param": 0.001,
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

    wandb.init(project='finetuner',
               entity='sig-net',
               config=config,
               name=config["model_id"])

    train_data, val_data = read_data_type(device=dev,
                                     experiment_id=config["experiment_id"],
                                     source=config["source"],
                                     type="large")

    trainer = FinetunerTrainer(iterations=config["iterations"],  # Passes through all dataset
                               train_data=train_data,
                               val_data=val_data,
                               num_classes=config["num_classes"],
                               fp_param=config["fp_param"],
                               fn_param=config["fn_param"],
                               device=device,
                               model_path=finetuner_path)

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_hidden_layers=config["num_hidden_layers"],
                                num_units=config["num_neurons"],
                                plot=True)
