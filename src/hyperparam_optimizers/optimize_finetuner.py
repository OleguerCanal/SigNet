import os
import sys

from skopt.space import Real, Integer, Categorical
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_data
from trainers.finetuner_trainer import FinetunerTrainer
from HyperParameterOptimizer.gaussian_process import GaussianProcessSearch


experiment_id = "finetuner_mixed_small_num_mut"
# experiment_id = "finetuner_mixed_large_num_mut"
iterations = 15
num_classes = 72
fp_param = 1e-3
fn_param = 1e-3

batch_sizes = Integer(name='batch_size', low=50, high=1000)
learning_rates = Real(name='lr', low=1e-5, high=5e-3)
num_units = Integer(name='num_units', low=50, high=800)
num_hidden_layers = Integer(name='num_hidden_layers', low=1, high=8)

input_file = None  # Use None to start from zero
# NOTE(claudia): Is this a problem if running it on multiple nodes of the cluster?
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
                               fp_param=fp_param,
                               fn_param=fn_param,
                               device=torch.device(dev))

    # Define hyperparameters to train
    search_space = [batch_sizes, learning_rates, num_hidden_layers, num_units]
    fixed_space = {"plot": True}

    def objective(batch_size,
                  lr,
                  num_hidden_layers,
                  num_units,
                  plot=False):
        config = {"batch_size": batch_size,
                "lr": lr,
                "num_hidden_layers": num_hidden_layers,
                "num_units": num_units}
        
        run = wandb.init(project='bayesian-' + experiment_id,
                         entity='sig-net',
                         config=config,
                         name=str(config))

        val = trainer.objective(batch_size=batch_size,
                                lr=lr,
                                num_hidden_layers=num_hidden_layers,
                                num_units=num_units,
                                plot=plot)
        wandb.log({"validation_score": val})
        run.finish()
        return val

    # Start optimization
    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=objective,
                                      input_file=input_file,
                                      output_file=output_file)  # Store tested points
    gp_search.init_session()
    best_metaparams, best_val = gp_search.get_maximum(
        n_calls=100,
        n_random_starts=20,
        noise=0.01,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)
