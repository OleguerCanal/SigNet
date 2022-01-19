
import collections
import copy
import os
import pathlib
import sys

import numpy as np
import pandas as pd

import torch
from torch import linalg
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.data_partitions import DataPartitions
from utilities.io import save_model, read_model
from utilities.metrics import distance_to_interval, get_pi_metrics
from models.error_finder import ErrorFinder
from loggers.error_finder_logger import ErrorFinderLogger
from modules.classified_tunning import ClassifiedFinetuner
from modules.combined_finetuner import CombinedFinetuner

class ErrorTrainer:
    def __init__(self,
                 iterations,
                 train_data,
                 val_data,
                 loging_path="../runs",
                 num_classes=72,
                 model_path=None,
                 device="cuda:0"):
        self.iterations = iterations  # Iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.device = device
        self.model_path = model_path
        self.logger = ErrorFinderLogger(
            path=None,
            experiment_id=None)

    def __loss(self,
               label,
               pred_lower,
               pred_upper):

        lagrange_missclassification = float(self.loss_params["lagrange_missclassification"])
        lagrange_pnorm = float(self.loss_params["lagrange_pnorm"])
        pnorm_order = int(self.loss_params["pnorm_order"] )
        lagrange_smalltozero = float(self.loss_params["lagrange_smalltozero"])
        pnorm_order = pnorm_order if pnorm_order%2 == 1 else pnorm_order + 1
        
        _EPS = 1e-6
        batch_size = float(pred_lower.shape[0])
        lower = label - pred_lower
        upper = pred_upper - label
        lower = nn.ReLU()(-lower)  # 0 if lower than label
        upper = nn.ReLU()(-upper)  # 0 if higher than label

        interval_length = ((pred_upper - pred_lower)**2)/batch_size
        
        # loss = interval_length + lagrange * missclassifications
        loss_by_mutation_signature =\
            interval_length +\
            lagrange_missclassification*(lower + upper)

        # p-norm by signature to avoid high errors
        loss_by_mutation = linalg.norm(lagrange_pnorm *\
            loss_by_mutation_signature, ord=pnorm_order, axis=1)
        loss = torch.mean(loss_by_mutation)

        # Send small to 0
        loss += lagrange_smalltozero*\
            torch.mean(torch.abs(pred_upper[label <= _EPS]))
        return loss

    def __meta_loss(self,
                    label,
                    pred_lower,
                    pred_upper):
        _EPS = 1e-6
        batch_size = float(pred_lower.shape[0])
        lower = label - pred_lower
        upper = pred_upper - label
        lower = nn.ReLU()(-lower)  # 0 if lower than label
        upper = nn.ReLU()(-upper)  # 0 if higher than label

        # Manage this case
        # assert(torch.isnan(pred_upper).any() == False)  # int len
        # assert(torch.isnan(pred_lower).any() == False)  # int len

        # Get interval length
        interval_length = torch.mean((pred_upper - pred_lower)**2)

        # If less than 95% in, add penalization
        # https://www.wolframalpha.com/input/?i=y+%3D+sigmoid%28%280.97-x%29*2000%29+from+0.95+to+1
        penalization = torch.mean((lower + upper < _EPS).to(torch.float32))
        # print("penalization:", penalization)
        # assert(torch.isnan(penalization).any() == False)  # pen 1
        penalization = nn.Sigmoid()((0.95 - penalization)*2000)
        # assert(torch.isnan(penalization).any() == False)  # pen 2
        meta_loss = interval_length + penalization
        return meta_loss

    def objective(self,
                  batch_size,
                  lr,
                  num_neurons_pos,
                  num_neurons_neg,
                  num_hidden_layers_pos,
                  num_hidden_layers_neg,
                  loss_params,
                  plot=False):
        print(batch_size, lr, num_neurons_pos, num_hidden_layers_pos,
              num_neurons_neg, num_hidden_layers_neg, loss_params)
        self.loss_params = loss_params
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        model = ErrorFinder(num_classes=self.num_classes,
                            num_hidden_layers_pos=int(num_hidden_layers_pos),
                            num_units_pos=int(num_neurons_pos),
                            num_hidden_layers_neg=int(num_hidden_layers_neg),
                            num_units_neg=int(num_neurons_neg))
        model.to(self.device)

        log_freq = 5
        if plot:
            wandb.watch(model, log_freq=log_freq, log_graph=True)

        optimizer = optim.Adam(model.parameters(),
                               lr=lr)
        meta_loss_vals = collections.deque(maxlen=20)
        max_found = -np.inf
        step = 0
        for _ in range(self.iterations):
            for _, train_label, train_weight_guess, num_mut, classification in tqdm(dataloader):
                optimizer.zero_grad()
                if "cuda" in self.device:
                    train_label.to(self.device)
                    train_weight_guess.to(self.device)
                    num_mut.to(self.device)
                    classification.to(self.device)
                train_pred_upper, train_pred_lower = model(weights=train_weight_guess,
                                                           num_mutations=num_mut,
                                                           classification=classification)

                # Compute loss
                train_loss = self.__loss(label=train_label,
                                         pred_lower=train_pred_lower,
                                         pred_upper=train_pred_upper)
                if torch.isnan(train_loss).any():
                    print("NANs appeared while training! Ending training now")
                    return -2.0
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    val_pred_upper, val_pred_lower = model(weights=self.val_dataset.prev_guess,
                                                           num_mutations=self.val_dataset.num_mut,
                                                           classification=self.val_dataset.classification)
                    val_loss = self.__loss(label=self.val_dataset.labels,
                                           pred_lower=val_pred_lower,
                                           pred_upper=val_pred_upper)
                    val_meta_loss = self.__meta_loss(label=self.val_dataset.labels,
                                                     pred_lower=val_pred_lower,
                                                     pred_upper=val_pred_upper)
                    meta_loss_vals.append(val_meta_loss.item())
                    max_found = max(max_found, -np.nanmean(meta_loss_vals))

                if plot and step % log_freq == 0:
                    pi_metrics_train = get_pi_metrics(train_label, train_pred_lower, train_pred_upper)
                    pi_metrics_val = get_pi_metrics(self.val_dataset.labels, val_pred_lower, val_pred_upper, collapse=False, dim=1)
                    self.logger.log(train_loss=train_loss,
                                    pi_metrics_train=pi_metrics_train,
                                    val_loss=val_loss,
                                    pi_metrics_val=pi_metrics_val,
                                    val_values_lower=val_pred_lower,
                                    val_values_upper=val_pred_upper,
                                    val_nummut=self.val_dataset.num_mut,
                                    step=step)
                del train_label, train_weight_guess, num_mut, classification, train_pred_upper, train_pred_lower, val_pred_upper, val_pred_lower
                if self.model_path is not None and step % 500 == 0:
                    save_model(model=model, directory=self.model_path)
                step += batch_size
        if self.model_path is not None:
            save_model(model=model, directory=self.model_path)
        return max_found


def train_errorfinder(config) -> float:
    from utilities.io import read_data
    from modules.combined_finetuner import baseline_guess_to_combined_finetuner_guess

    # Select training device
    dev = "cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    print("Using device:", device)

    # Set paths
    errorfinder_path = os.path.join(config["models_dir"], config["model_id"])
    classifier_path = os.path.join(config["models_dir"], config["classifier_id"])
    finetuner_realistic_low_path = os.path.join(config["models_dir"], config["finetuner_realistic_low_id"])
    finetuner_realistic_large_path = os.path.join(config["models_dir"], config["finetuner_realistic_large_id"])

    if config["enable_logging"]:
        wandb.init(project=config["wandb_project_id"],
                entity='sig-net',
                config=config,
                name=config["model_id"])

    # Load data
    train_real_low, val_real_low = read_data(experiment_id=config["data_id"],
                                            source="generator_low",
                                            device="cpu")
    train_real_large, val_real_large = read_data(experiment_id=config["data_id"],
                                                source="generator_large",
                                                device="cpu")
    train_perturbed_low, val_perturbed_low = read_data(experiment_id=config["data_id"],
                                            source="perturbed_low",
                                            device="cpu")
    train_perturbed_large, val_perturbed_large = read_data(experiment_id=config["data_id"],
                                                source="perturbed_large",
                                                device="cpu")
    train_data = train_real_low
    train_data.append(train_real_large)
    train_data.append(train_perturbed_low)
    train_data.append(train_perturbed_large)
    train_data.perm()

    del train_real_low
    del train_real_large
    del train_perturbed_low
    del train_perturbed_large

    val_data = val_real_low
    val_data.append(val_real_large)
    val_data.append(val_perturbed_low)
    val_data.append(val_perturbed_large)

    del val_real_low
    del val_real_large
    del val_perturbed_low
    del val_perturbed_large

    # model = ClassifiedFinetuner(classifier=read_model(classifier_path)
    model = CombinedFinetuner(low_mum_mut_dir=finetuner_realistic_low_path,
                              large_mum_mut_dir=finetuner_realistic_large_path)
    classifier = read_model(classifier_path)

    train_data = baseline_guess_to_combined_finetuner_guess(model=model,
                                                            classifier=classifier,
                                                            data=train_data)
    val_data = baseline_guess_to_combined_finetuner_guess(model=model,
                                                          classifier=classifier,
                                                          data=val_data)

    # train_data.to(device=device)  # We keep the train data in cpu to save memory
    val_data.to(device=device)  # If still a problem we can make the val_data smaller
    torch.cuda.empty_cache()

    trainer = ErrorTrainer(iterations=config["iterations"],  # Passes through all dataset
                           train_data=train_data,
                           val_data=val_data,
                           num_classes=config["num_classes"],
                           device=device,
                           model_path=errorfinder_path)

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_neurons_pos=config["num_neurons_pos"],
                                num_neurons_neg=config["num_neurons_neg"],
                                num_hidden_layers_pos=config["num_hidden_layers_pos"],
                                num_hidden_layers_neg=config["num_hidden_layers_neg"],
                                loss_params=config["loss_params"],
                                plot=config["enable_logging"])
    return min_val