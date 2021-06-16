import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_data
from trainers.finetuner_trainer import FinetunerTrainer

experiment_id = "finetuner_model"
model_path = "../trained_models"
iterations = 7
num_classes = 72
fp_param = 1
fn_param = 1

batch_size = 336
lr = 0.001
num_hidden_layers = 1
num_neurons = 1500

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = "cpu"
    device = torch.device(dev)
    print("Using device:", dev)

    train_input, train_guess_0, train_label, val_input, val_guess_0, val_label = read_data(
        dev)

    trainer = FinetunerTrainer(iterations=iterations,  # Passes through all dataset
                               train_input=train_input,
                               train_weight_guess=train_guess_0,
                               train_label=train_label,
                               val_input=val_input,
                               val_weight_guess=val_guess_0,
                               val_label=val_label,
                               experiment_id=experiment_id,
                               num_classes=num_classes,
                               fp_param=fp_param,
                               fn_param=fn_param,
                               device=device,
                               model_path=model_path)

    min_val = trainer.objective(batch_size=batch_size,
                                lr=lr,
                                num_hidden_layers=num_hidden_layers,
                                num_units=num_neurons,
                                plot=True)
