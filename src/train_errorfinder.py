import gc
import os
import sys

import torch
from torchsummary import summary

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.finetuner import FineTuner, baseline_guess_to_finetuner_guess
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data

source = "mixed"
experiment_id = "exp_0"
models_path = os.path.join("../trained_models", experiment_id)
errorfinder_path = os.path.join(models_path, "errorfinder_" + source)
iterations = 40
num_classes = 72

# Error finder params
batch_size = 500
lr = 0.0001
num_hidden_layers_pos = 2
num_neurons_pos = 1000
num_hidden_layers_neg = 2
num_neurons_neg = 1000

# Finetuner args
finetuner_path = os.path.join(models_path, "finetuner_" + source)
fintuner_args = {
    "num_hidden_layers": 3,
    "num_units": 600
}

if __name__ == "__main__":
    # Select training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load data
    train_data, val_data = read_data(experiment_id=experiment_id,
                                     source=source,
                                     device="cpu")

    train_data = baseline_guess_to_finetuner_guess(finetuner_args=fintuner_args,
                                                   trained_finetuner_file=finetuner_path,
                                                   data=train_data)
    val_data = baseline_guess_to_finetuner_guess(finetuner_args=fintuner_args,
                                                 trained_finetuner_file=finetuner_path,
                                                 data=val_data)

    train_data.to(device=device)
    val_data.to(device=device)
    torch.cuda.empty_cache()

    trainer = ErrorTrainer(iterations=iterations,  # Passes through all dataset
                           train_data=train_data,
                           val_data=val_data,
                           num_classes=num_classes,
                           device=device,
                           model_path=errorfinder_path)

    min_val = trainer.objective(batch_size=batch_size,
                                lr=lr,
                                num_neurons_pos=num_neurons_pos,
                                num_neurons_neg=num_neurons_neg,
                                num_hidden_layers_pos=num_hidden_layers_pos,
                                num_hidden_layers_neg=num_hidden_layers_neg,
                                plot=True)
