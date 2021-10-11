import gc
import os
import sys

from skopt.space import Real, Integer, Categorical
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer.gaussian_process import GaussianProcessSearch
from models.finetuner import FineTuner, baseline_guess_to_finetuner_guess
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data

source = "random"
experiment_id = "exp_0"
iterations = 8
model_path = "../../trained_models/" + experiment_id
num_classes = 72

batch_sizes = Integer(name='batch_size', low=50, high=1000)
learning_rates = Real(name='lr', low=0.00001, high=0.05)
neurons_pos = Integer(name='num_neurons_pos', low=20, high=1500)
layers_pos = Integer(name='num_hidden_layers_pos', low=1, high=10)
neurons_neg = Integer(name='num_neurons_neg', low=20, high=1500)
layers_neg = Integer(name='num_hidden_layers_neg', low=1, high=10)

input_file = None
output_file = "search_results_" + experiment_id + ".csv"

# Finetuner params
finetuner_path = os.path.join(model_path, "finetuner_" + source)

if __name__ == "__main__":
# Select training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load data
    train_data, val_data = read_data(experiment_id=experiment_id,
                                     source=source,
                                     device="cpu",
                                     data_folder="../../data/")

    train_data = baseline_guess_to_finetuner_guess(trained_finetuner_dir=finetuner_path,
                                                   data=train_data)
    val_data = baseline_guess_to_finetuner_guess(trained_finetuner_dir=finetuner_path,
                                                 data=val_data)

    train_data.to(device=device)
    val_data.to(device=device)
    torch.cuda.empty_cache()

    trainer = ErrorTrainer(iterations=iterations,  # Passes through all dataset
                           train_data=train_data,
                           val_data=val_data,
                           experiment_id=experiment_id,
                           num_classes=num_classes,
                           device=device,)


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
        n_calls=1000,
        n_random_starts=100,
        noise=0.01,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)
