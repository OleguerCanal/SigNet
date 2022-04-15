import os
import sys

from skopt.space import Real, Integer, Categorical
import torch
import wandb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import read_config, read_data
from trainers.finetuner_trainer import FinetunerTrainer
from HyperParameterOptimizer.src.gaussian_process_search import GaussianProcessSearch


experiment_id = "finetuner_oversample_low"
num_classes = 72

batch_sizes = Integer(name='batch_size', low=50, high=500)
learning_rates = Real(name='lr', low=1e-5, high=5e-3)
num_units = Integer(name='num_units', low=50, high=800)
num_hidden_layers = Integer(name='num_hidden_layers', low=1, high=6)

input_file = None  # Use None to start from zero
output_file = "search_results_" + experiment_id + ".csv"

if __name__ == "__main__":
    # Read base config
    config = read_config(path="../configs/finetuner/finetuner_nobaseline_low.yaml")
    iterations = config['iterations']

    # Read data
    train_data, val_data = read_data(device=config['device'],
                                     experiment_id=config['data_id'],
                                     source=config['source'],
                                     data_folder="../../data/")

    # Instantiate Finetuner trainer
    trainer = FinetunerTrainer(iterations=iterations,  # Passes through all dataset
                               train_data=train_data,
                               val_data=val_data,
                               network_type=config['source'],
                               device=torch.device(config['device']),
                               sigmoid_params=config['sigmoid_params'],
                               model_path=config['models_dir'])

    # Define hyperparameters to train
    search_space = [batch_sizes, learning_rates, num_hidden_layers, num_units]
    fixed_space = {"plot": True}

    def objective(batch_size,
                  lr,
                  num_hidden_layers,
                  num_units,
                  plot=False):
        
        run_config = {
            "batch_size": batch_size,
            "lr": lr,
            "num_hidden_layers": num_hidden_layers,
            "num_units": num_units,
            }

        config.update(run_config)

        run = wandb.init(project='bayesian_' + experiment_id,
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
        n_calls=1000,
        n_random_starts=50,
        noise=0.01,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)
