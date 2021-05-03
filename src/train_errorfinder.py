import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.finetuner import FineTuner
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data


experiment_id = "error_learner_model_1"
model_path = "../trained_models"
finetuner_model_name = "finetuner_model_1"
iterations = 7
num_classes = 72

# Error finder params
batch_size = 500
lr = 0.0001
num_hidden_layers_pos = 1
num_neurons_pos = 500
num_hidden_layers_neg = 1
num_neurons_neg = 500
normalize_mut = 2e4

# Finetuner params
num_hidden_layers = 1
num_units = 1500

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev = "cpu"
    device = torch.device(dev)
    print("Using device:", dev)

    train_input, train_guess_0, train_label, val_input, val_guess_0, val_label = read_data(
        dev)

    finetuner = FineTuner(num_classes=72,
                          num_hidden_layers=num_hidden_layers,
                          num_units=num_units)
    finetuner.to(device)
    finetuner.load_state_dict(torch.load(os.path.join(model_path, finetuner_model_name), map_location=torch.device('cpu')))
    finetuner.eval()
    train_guess_1 = finetuner(mutation_dist=train_input,
                              weights=train_guess_0)

    val_guess_1 = finetuner(mutation_dist=val_input,
                            weights=val_guess_0)
    del finetuner
    del train_guess_0
    del val_guess_0

    trainer = ErrorTrainer(iterations=iterations,  # Passes through all dataset
                           train_input=train_input,
                           train_weight_guess=train_guess_1,
                           train_label=train_label,
                           val_input=val_input,
                           val_weight_guess=val_guess_1,
                           val_label=val_label,
                           experiment_id=experiment_id,
                           num_classes=num_classes,
                           device=device,
                           model_path=model_path)

    min_val = trainer.objective(batch_size=batch_size,
                                lr=lr,
                                num_neurons_pos=num_neurons_pos,
                                num_neurons_neg=num_neurons_neg,
                                num_hidden_layers_pos=num_hidden_layers_pos,
                                num_hidden_layers_neg=num_hidden_layers_neg,
                                normalize_mut=normalize_mut,
                                plot=True)
