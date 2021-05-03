import os
import sys

from skopt.space import Real, Integer, Categorical
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer.gaussian_process import GaussianProcessSearch
from models.finetuner import FineTuner 
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data

experiment_id = "error_learner_0"
iterations = 5
num_classes = 72

batch_sizes = Integer(name='batch_size', low=50, high=1000)
learning_rates = Real(name='lr', low=0.00001, high=0.005)
neurons_pos = Integer(name='num_neurons_pos', low=20, high=1500)
layers_pos = Integer(name='num_hidden_layers_pos', low=1, high=10)
neurons_neg = Integer(name='num_neurons_neg', low=20, high=1500)
layers_neg = Integer(name='num_hidden_layers_neg', low=1, high=10)
normalize_mut_param = Integer(name='normalize_mut', low=1e4, high=1e6)

input_file = None
output_file = "search_results_" + experiment_id + ".csv"

# Finetuner params
finetuner_model_name = "test_0"
num_hidden_layers = 1
num_units = 1500
model_path = "../../trained_models"

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dev = "cpu"
    device = torch.device(dev)
    print("Using device:", dev)

    train_input, train_guess_0, train_label, val_input, val_guess_0, val_label = read_data(dev, data_folder="../../data")

    finetuner = FineTuner(num_classes=72,
                          num_hidden_layers=num_hidden_layers,
                          num_units=num_units)
    finetuner.to(device)
    finetuner.load_state_dict(torch.load(os.path.join(model_path, finetuner_model_name)))
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
                               device=device)


    search_space = [batch_sizes, learning_rates, neurons_pos,
                    layers_pos, neurons_neg, layers_neg, normalize_mut_param]
    fixed_space = {"plot": False}

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=trainer.objective,
                                      input_file=input_file,  # Use None to start from zero
                                      output_file=output_file)  # Store tested points
    gp_search.init_session()
    best_metaparams, best_val = gp_search.get_maximum(
        n_calls=100,
        n_random_starts=0,
        noise=0.01,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)