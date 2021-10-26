import gc
import os
import sys

from skopt.space import Real, Integer, Categorical
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer import GaussianProcessSearch, ParallelSearcher
from errorfinder_job_instance import ErrorfinderJobInstance

experiment_id = "errorfinder_parallel"

# Training params
batch_sizes = Integer(name='batch_size', low=50, high=1000)
learning_rates = Real(name='lr', low=1e-5, high=1e-3)
neurons_pos = Integer(name='num_neurons_pos', low=20, high=1500)
layers_pos = Integer(name='num_hidden_layers_pos', low=1, high=10)
neurons_neg = Integer(name='num_neurons_neg', low=20, high=1500)
layers_neg = Integer(name='num_hidden_layers_neg', low=1, high=10)
# Loss params
lagrange_missclassification_param = Real(name="lagrange_missclassification", low=5e-3, high=1e-2)
lagrange_pnorm_param = Real(name="lagrange_pnorm", low=1e3, high=1e5)
lagrange_smalltozero_param = Real(name="lagrange_smalltozero", low=0.1, high=10)
pnorm_order_param = Integer(name="pnorm_order", low=3, high=9)

input_file = None
output_file = "search_results_" + experiment_id + ".csv"

if __name__ == "__main__":
    search_space = [batch_sizes, learning_rates, neurons_pos,
                    layers_pos, neurons_neg, layers_neg,
                    lagrange_missclassification_param,
                    pnorm_order_param, lagrange_smalltozero_param,
                    lagrange_pnorm_param]
    fixed_space = {"plot": True}

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=None,
                                      input_file=input_file,  # Use None to start from zero
                                      output_file=output_file)  # Store tested points
    gp_search.init_session()

    searcher = ParallelSearcher(optimizer=gp_search, job_class=ErrorfinderJobInstance)
    searcher.optimize(
        n_calls=500,
        n_random_starts=100,
        noise=0.01,
        n_parallel_jobs=5,
        verbose=True,
        plot_results=True)