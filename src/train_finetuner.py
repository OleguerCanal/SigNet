from trainers.finetuner_trainer import FinetunerTrainer
from utilities.io import read_data
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

experiment_id = "finetuner_mixed"
model_path = "../trained_models/exp_0/"
iterations = 20
num_classes = 72
fp_param = 0.001
fn_param = 0.001

batch_size = 500
lr = 0.0001
num_hidden_layers = 2
num_neurons = 1300

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # dev = "cuda"
    device = torch.device(dev)
    print("Using device:", dev)

    train_input, train_baseline, train_label,\
        val_input, val_baseline, val_label = read_data(device=dev,
                                                       experiment_id="exp_0",
                                                       source="mixed")

    trainer = FinetunerTrainer(iterations=iterations,  # Passes through all dataset
                               train_input=train_input,
                               train_weight_guess=train_baseline,
                               train_label=train_label,
                               val_input=val_input,
                               val_weight_guess=val_baseline,
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
