import gc
import os
import sys

import torch
from torchsummary import summary

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.finetuner import FineTuner
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data

source = "realistic"
experiment_id = "exp_0"
model_path = "../trained_models/" + experiment_id
errorfinder_model_name = "errorfinder_2"
finetuner_model_name = "finetuner_" + source
iterations = 30
num_classes = 72

# Error finder params
batch_size = 500
lr = 0.0001
num_hidden_layers_pos = 1
num_neurons_pos = 1000
num_hidden_layers_neg = 1
num_neurons_neg = 1000

# Finetuner params
num_hidden_layers = 2
num_units = 1300


if __name__ == "__main__":
    # dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev = "cpu"
    device = torch.device(dev)
    print("Using device:", dev)

    train_input, train_baseline, train_label,\
        val_input, val_baseline, val_label = read_data(device=dev,
                                                       experiment_id=experiment_id,
                                                       source=source)
    
    finetuner = FineTuner(num_classes=72,
                          num_hidden_layers=num_hidden_layers,
                          num_units=num_units)
    finetuner.load_state_dict(torch.load(os.path.join(model_path, finetuner_model_name),  map_location=torch.device('cpu')))
    finetuner.to("cpu")
    finetuner.eval()
    
    with torch.no_grad():
        train_guess_1 = finetuner(mutation_dist=train_input,
                                weights=train_baseline,
                                num_mut=train_label[:, -1].reshape(-1, 1))

        val_guess_1 = finetuner(mutation_dist=val_input,
                                weights=val_baseline,
                                num_mut=val_label[:, -1].reshape(-1, 1))

    del finetuner
    del train_baseline
    del val_baseline
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
                           experiment_id=errorfinder_model_name,
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
