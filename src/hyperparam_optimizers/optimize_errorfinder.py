import gc
import os
import sys

from skopt.space import Real, Integer, Categorical
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer.gaussian_process import GaussianProcessSearch
from models.finetuner import FineTuner, baseline_guess_to_finetuner_guess
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data

experiment_id = "errorfinder_realistic_smallnummut"
source = "realistic"
data_experiment_id = "exp_0"
finetuner_model_id = "exp_2_nets/finetuner_realistic"
iterations = 3
num_classes = 72

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

# Finetuner params
models_path = "../../trained_models"
finetuner_path = os.path.join(models_path, finetuner_model_id)

if __name__ == "__main__":
# Select training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load data
    train_data, val_data = read_data(experiment_id=data_experiment_id,
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
                           model_path=None,
                           num_classes=num_classes,
                           device=device,)

    search_space = [batch_sizes, learning_rates, neurons_pos,
                    layers_pos, neurons_neg, layers_neg,
                    lagrange_missclassification_param,
                    pnorm_order_param, lagrange_smalltozero_param,
                    lagrange_pnorm_param]
    fixed_space = {"plot": True}

    def objective(batch_size,
                  lr,
                  num_neurons_pos,
                  num_hidden_layers_pos,
                  num_neurons_neg,
                  num_hidden_layers_neg,
                  lagrange_missclassification,
                  lagrange_pnorm,
                  lagrange_smalltozero,
                  pnorm_order,
                  plot=False):

        loss_params = {
            "lagrange_missclassification": lagrange_missclassification ,
            "lagrange_pnorm": lagrange_pnorm,
            "lagrange_smalltozero": lagrange_smalltozero,
            "pnorm_order": pnorm_order,
        }

        config = {"batch_size": batch_size,
                  "lr": lr,
                  "num_neurons_pos": num_neurons_pos,
                  "num_hidden_layers_pos": num_hidden_layers_pos,
                  "num_neurons_neg": num_neurons_neg,
                  "num_hidden_layers_neg": num_hidden_layers_neg,
                  "loss_params": loss_params
                  }
        if plot:
            run = wandb.init(project='bayesian-' + experiment_id,
                            entity='sig-net',
                            config=config,
                            name=str(config))

        val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_neurons_pos=config["num_neurons_pos"],
                                num_neurons_neg=config["num_neurons_neg"],
                                num_hidden_layers_pos=config["num_hidden_layers_pos"],
                                num_hidden_layers_neg=config["num_hidden_layers_neg"],
                                loss_params=config["loss_params"],
                                plot=plot)
        if plot:
            wandb.log({"validation_score": val})
            run.finish()
        return val

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=objective,
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
