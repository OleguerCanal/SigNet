import os
import sys

from skopt.space import Real, Integer, Categorical
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_data
from trainers.finetuner_trainer import FinetunerTrainer
from HyperParameterOptimizer.gaussian_process import GaussianProcessSearch

experiment_id = "finetuner_0"
iterations = 5
num_classes = 72
fp_param = 1e-3
fn_param = 1e-3

batch_sizes = Integer(name='batch_size', low=50, high=1000)
learning_rates = Real(name='lr', low=0.00001, high=0.005)
num_units = Integer(name='num_units', low=20, high=700)
num_hidden_layers = Integer(name='num_hidden_layers', low=1, high=10)

input_file = None  # Use None to start from zero
output_file = "search_results_" + experiment_id + ".csv"

if __name__ == "__main__":
    # Select training device
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", dev)

    # Read data
    train_data, val_data = read_data(device=dev,
                                     experiment_id="exp_0",
                                     source="random",
                                     data_folder="../../data/")

    # Instantiate Finetuner trainer
    trainer = FinetunerTrainer(iterations=iterations,  # Passes through all dataset
                               train_data=train_data,
                               val_data=val_data,
                               experiment_id=experiment_id,
                               num_classes=num_classes,
                               fp_param=fp_param,
                               fn_param=fn_param,
                               device=torch.device(dev))
    
    # Define hyperparameters to train
    search_space = [batch_sizes, learning_rates, num_units, num_hidden_layers]
    fixed_space = {"plot": False}
    
    # Start optimization
    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=trainer.objective,
                                      input_file=input_file,
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
