import gc
import os
import sys

import torch
from torchsummary import summary

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.finetuner import FineTuner
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data


experiment_id = "error_finder_optimized"
model_path = "../trained_models"
finetuner_model_name = "finetuner_model_optimized"
iterations = 8
num_classes = 72

# Error finder params
batch_size = 1000
lr = 0.001
num_hidden_layers_pos = 3
num_neurons_pos = 700
num_hidden_layers_neg = 2
num_neurons_neg = 800

# Finetuner params
num_hidden_layers = 1
num_units = 1300

def print_size(name, tensor):
    size = tensor.element_size() * tensor.nelement() * 1e-6
    print(name, "has size:", size, "MB (", tensor.type(), ")")

if __name__ == "__main__":
    # dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev = "cpu"
    device = torch.device(dev)
    print("Using device:", dev)

    train_input, train_guess_0, train_label, val_input, val_guess_0, val_label = read_data("cpu")
    
    finetuner = FineTuner(num_classes=72,
                          num_hidden_layers=num_hidden_layers,
                          num_units=num_units)
    finetuner.load_state_dict(torch.load(os.path.join(model_path, finetuner_model_name),  map_location=torch.device('cpu')))
    finetuner.to("cpu")
    finetuner.eval()
    
    with torch.no_grad():
        train_guess_1 = finetuner(mutation_dist=train_input,
                                weights=train_guess_0)

        val_guess_1 = finetuner(mutation_dist=val_input,
                                weights=val_guess_0)

    del finetuner
    del train_guess_0
    del val_guess_0
    gc.collect()
    torch.cuda.empty_cache()

    train_input = train_input.to(device)
    train_guess_1 = train_guess_1.to(device)
    train_label = train_label.to(device)
    val_input = val_input.to(device)
    val_guess_1 = val_guess_1.to(device)
    val_label = val_label.to(device)

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
                                plot=True)
