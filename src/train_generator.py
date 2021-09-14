
import os
import sys

import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.yapsa_inspired_baseline import YapsaInspiredBaseline
from models.finetuner import FineTuner
from utilities.io import read_data, read_real_data, read_signatures
from trainers.generator_trainer import GeneratorTrainer

config = {
    # IDs
    "experiment_id": "exp_real_data",
    "model_id": "real data generator",

    # Training params
    "source": "real",
    "iterations": 40,
    "num_classes": 72,
    "batch_size": 500,
    "lr": 0.0001,

    # Network params
    "num_hidden_layers": 3,
    "num_neurons": 600,
}

models_path = os.path.join("../trained_models", config["experiment_id"])
finetuner_path = os.path.join("../trained_models/exp_0", "finetuner_random")
generator_path = os.path.join(models_path, "generator_" + config["source"])

if __name__ == "__main__":
    # Select training device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print("Using device:", device)

    wandb.init(project='generator',
               entity='sig-net',
               config=config)

    # Load data
    signatures = read_signatures("../data/data.xlsx")
    train_data = read_real_data(experiment_id=config["experiment_id"],
                                     device=device)
    train_data.to(device=device)
    # val_data.to(device=device)

    baseline = YapsaInspiredBaseline(signatures=signatures)

    finetuner = FineTuner(num_hidden_layers=config["num_hidden_layers"],
                          num_units=config["num_neurons"])
    finetuner.load_state_dict(torch.load(
        finetuner_path, map_location=torch.device('cpu')))
    finetuner.to(device)
    finetuner.eval()

    trainer = GeneratorTrainer(
        signatures=signatures,
        baseline=baseline,
        finetuner=finetuner,
        iterations=config["iterations"],  # Passes through all dataset
        train_data=train_data,
        #val_data=val_data,
        model_path=generator_path)

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                plot=True)
