import gc
import os
import sys

from skopt.space import Real, Integer, Categorical
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer import GaussianProcessSearch, ParallelSearcher
from finetuner_job_instance import FinetunerJobInstance

experiment_id = "finetuner_realistic"

# Training params
batch_sizes = Integer(name='batch_size', low=50, high=1000)
learning_rates = Real(name='lr', low=1e-5, high=5e-3)
neurons = Integer(name='num_neurons', low=50, high=800)
layers = Integer(name='num_hidden_layers', low=1, high=8)

input_file = "search_results/search_results_" + experiment_id + ".csv"
output_file = "search_results/search_results_" + experiment_id + "_final.csv"

if __name__ == "__main__":
    search_space = [batch_sizes, learning_rates, neurons,
                    layers]
    fixed_space = {"plot": True}

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=None,
                                      input_file=input_file,  # Use None to start from zero
                                      output_file=output_file)  # Store tested points
    gp_search.init_session()

    searcher = ParallelSearcher(optimizer=gp_search, job_class=FinetunerJobInstance)
    searcher.optimize(
        n_calls=500,
        n_random_starts=5,
        noise=0.01,
        n_parallel_jobs=5,
        verbose=True,
        plot_results=True)