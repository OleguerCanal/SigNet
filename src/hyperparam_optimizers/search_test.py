import os
import sys

from skopt.space import Real, Integer, Categorical
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer import GaussianProcessSearch, ParallelSearcher
from test_job_instance import TestJobInstance

experiment_id = "test"

# Search space
batch_sizes = Integer(name='batch_size', low=10, high=5000)
learning_rates = Real(name='lr', low=1e-5, high=1e-3)
num_units = Integer(name='num_units', low=50, high=1000)
num_hidden_layers = Integer(name='num_hidden_layers', low=1, high=8)

input_file = None  # Use None to start from zero
output_file = "search_results_" + experiment_id + ".csv"

if __name__ == "__main__":
    # Define hyperparameters to train
    search_space = [batch_sizes, learning_rates, num_hidden_layers, num_units]
    fixed_space = {"plot": True}


    # Instantiate optimizer
    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=None,
                                      input_file=input_file,
                                      output_file=output_file)  # Store tested points
    gp_search.init_session()


    searcher = ParallelSearcher(optimizer=gp_search, job_class=TestJobInstance)
    searcher.optimize(
        n_calls=200,
        n_random_starts=50,
        noise=0.01,
        n_parallel_jobs=3,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)
