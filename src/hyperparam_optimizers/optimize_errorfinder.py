import gc
import os
import sys

from skopt.space import Real, Integer, Categorical
import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer import GaussianProcessSearch, ParallelSearcher
from models.finetuner import FineTuner, baseline_guess_to_finetuner_guess
from trainers.error_trainer import ErrorTrainer
from utilities.io import read_data, read_model
from modules.combined_finetuner import CombinedFinetuner, baseline_guess_to_combined_finetuner_guess

experiment_id = "errorfinder_bayesian"
data_experiment_id = "exp_oversample"
finetuner_low_path = "../../trained_models/exp_all/finetuner_low"
finetuner_large_path = "../../trained_models/exp_all/finetuner_large"
classifier_path = "../../trained_models/exp_all/classifier"
iterations = 3
num_classes = 72

# Training params
batch_sizes = Integer(name='batch_size', low=20, high=500)
learning_rates = Real(name='lr', low=1e-5, high=1e-3)
neurons_pos = Integer(name='num_neurons_pos', low=50, high=1200)
layers_pos = Integer(name='num_hidden_layers_pos', low=1, high=6)
neurons_neg = Integer(name='num_neurons_neg', low=50, high=1200)
layers_neg = Integer(name='num_hidden_layers_neg', low=1, high=6)

# Loss params
lagrange_missclassification_vector_param = [Real(name="lagrange_missclassification_" + str(i), low=8e-3, high=1e-1) for i in range(72)]
lagrange_pnorm_param = Real(name="lagrange_pnorm", low=1e3, high=1e5)
lagrange_smalltozero_param = Real(name="lagrange_smalltozero", low=0.1, high=10)
pnorm_order_param = Integer(name="pnorm_order", low=3, high=9)

input_file = None
output_file = "search_results_" + experiment_id + ".csv"

if __name__ == "__main__":
# Select training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load data
    train_real_low, val_real_low = read_data(experiment_id=data_experiment_id,
                                            source="low",
                                            data_folder="../../data",
                                            device=device)
    train_real_large, val_real_large = read_data(experiment_id=data_experiment_id,
                                                source="large",
                                                data_folder="../../data",
                                                device=device)
    train_data = train_real_low
    train_data.append(train_real_large)
    train_data.perm()

    del train_real_low
    del train_real_large

    val_data = val_real_low
    val_data.append(val_real_large)

    del val_real_low
    del val_real_large

    classifier = read_model(classifier_path)    
    finetuner = CombinedFinetuner(low_mum_mut_dir=finetuner_low_path,
                              large_mum_mut_dir=finetuner_large_path)

    train_data = baseline_guess_to_combined_finetuner_guess(model=finetuner,
                                                            classifier=classifier,
                                                            data=train_data)
    val_data = baseline_guess_to_combined_finetuner_guess(model=finetuner,
                                                          classifier=classifier,
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
                    pnorm_order_param, lagrange_smalltozero_param,
                    lagrange_pnorm_param] + lagrange_missclassification_vector_param
    fixed_space = {"plot": True}

    def objective(**kwargs):

        lagrange_missclassification_vector = [kwargs["lagrange_missclassification_"+str(i)] for i in range(72)]
        loss_params = {
            "lagrange_missclassification_vector":  lagrange_missclassification_vector,
            "lagrange_pnorm": kwargs["lagrange_pnorm"],
            "lagrange_smalltozero": kwargs["lagrange_smalltozero"],
            "pnorm_order": kwargs["pnorm_order"],
        }

        config = {"batch_size": kwargs["batch_size"],
                  "lr": kwargs["lr"],
                  "num_neurons_pos": kwargs["num_neurons_pos"],
                  "num_hidden_layers_pos": kwargs["num_hidden_layers_pos"],
                  "num_neurons_neg": kwargs["num_neurons_neg"],
                  "num_hidden_layers_neg": kwargs["num_hidden_layers_neg"],
                  "loss_params": loss_params
                  }
        # if plot:
        #     run = wandb.init(project='bayesian-' + experiment_id,
        #                     entity='sig-net',
        #                     config=config,
        #                     name=str(config))

        val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_neurons_pos=config["num_neurons_pos"],
                                num_neurons_neg=config["num_neurons_neg"],
                                num_hidden_layers_pos=config["num_hidden_layers_pos"],
                                num_hidden_layers_neg=config["num_hidden_layers_neg"],
                                loss_params=config["loss_params"],
                                plot=False)
        # if plot:
        #     wandb.log({"validation_score": val})
        #     run.finish()
        return val

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=objective,
                                      input_file=input_file,  # Use None to start from zero
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
