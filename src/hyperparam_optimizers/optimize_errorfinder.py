import gc
import os
import sys

from skopt.space import Real, Integer, Categorical
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer.gaussian_process import GaussianProcessSearch
from models.finetuner import FineTuner 
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data

experiment_id = "all_final"
iterations = 8
num_classes = 72

batch_sizes = Integer(name='batch_size', low=50, high=1000)
learning_rates = Real(name='lr', low=0.000001, high=0.01)
neurons_pos = Integer(name='num_neurons_pos', low=20, high=1500)
layers_pos = Integer(name='num_hidden_layers_pos', low=1, high=10)
neurons_neg = Integer(name='num_neurons_neg', low=20, high=1500)
layers_neg = Integer(name='num_hidden_layers_neg', low=1, high=10)

input_file = "errorfinder_opt/search_results_all.csv"
output_file = "errorfinder_opt/search_results_" + experiment_id + ".csv"

# Finetuner params
finetuner_model_name = "finetuner_model_optimized"
num_hidden_layers = 1
num_units = 1300
model_path = "../../trained_models"

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    print("Using device:", dev)

    train_input, train_guess_0, train_label, val_input, val_guess_0, val_label = read_data("cpu", data_folder="../../data")

    finetuner = FineTuner(num_classes=72,
                          num_hidden_layers=num_hidden_layers,
                          num_units=num_units)
    finetuner.load_state_dict(torch.load(os.path.join(model_path, finetuner_model_name), map_location=torch.device('cpu')))
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


    search_space = [batch_sizes, learning_rates, neurons_pos,
                    layers_pos, neurons_neg, layers_neg]
    fixed_space = {"plot": False}

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=trainer.objective,
                                      input_file=input_file,  # Use None to start from zero
                                      output_file=output_file)  # Store tested points
    gp_search.init_session()
    best_metaparams, best_val = gp_search.get_maximum(
        n_calls=500,
        n_random_starts=1,
        noise=0.01,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)
