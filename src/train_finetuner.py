import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.finetuner_trainer import FinetunerTrainer
from utilities.io import read_data

experiment_id = "finetuner_random"
model_path = "../trained_models/exp_0/"
iterations = 20
num_classes = 72
fp_param = 0.001
fn_param = 0.001

batch_size = 500
lr = 0.0001
num_hidden_layers = 2
num_neurons = 600

if __name__ == "__main__":
    # dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev = "cuda"
    device = torch.device(dev)
    print("Using device:", dev)

    train_data, val_data = read_data(device=dev,
                                     experiment_id="exp_0",
                                     source="random")

    trainer = FinetunerTrainer(iterations=iterations,  # Passes through all dataset
                               train_data=train_data,
                               val_data=val_data,
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
