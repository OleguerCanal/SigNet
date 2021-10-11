
import collections
import copy
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.data_partitions import DataPartitions
from utilities.io import save_model
from utilities.metrics import distance_to_interval, get_pi_metrics
from models.error_finder import ErrorFinder
from loggers.error_finder_logger import ErrorFinderLogger

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

        lagrange_missclassification = self.loss_params["lagrange_missclassification"]
        lagrange_pnorm = self.loss_params["lagrange_pnorm"]
        pnorm_order = self.loss_params["pnorm_order"] 
        lagrange_smalltozero = self.loss_params["lagrange_smalltozero"]
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
        loss_by_mutation = torch.linalg.norm(lagrange_pnorm *\
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

        # Get interval length
        interval_length = torch.mean((pred_upper - pred_lower)**2)

        # If less than 97% in, add penalisation
        # https://www.wolframalpha.com/input/?i=y+%3D+sigmoid%28%280.97-x%29*2000%29+from+0.95+to+1
        penalization = torch.mean((lower + upper > 0).to(torch.float32))
        penalization = nn.Sigmoid()((0.97 - penalization)*2000)
        meta_loss = interval_length + penalization
        assert (torch.isnan(meta_loss).any() == False)
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
        meta_loss_vals = collections.deque(maxlen=100)
        max_found = -np.inf
        step = 0
        for _ in range(self.iterations):
            for _, train_label, train_weight_guess, num_mut in tqdm(dataloader):
                optimizer.zero_grad()
                train_pred_upper, train_pred_lower = model(weights=train_weight_guess,
                                                           num_mutations=num_mut)

                # Compute loss
                train_loss = self.__loss(label=train_label,
                                         pred_lower=train_pred_lower,
                                         pred_upper=train_pred_upper)
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    val_pred_upper, val_pred_lower = model(weights=self.val_dataset.prev_guess,
                                                           num_mutations=self.val_dataset.num_mut)
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
                    pi_metrics_val = get_pi_metrics(self.val_dataset.labels, val_pred_lower, val_pred_upper)
                    self.logger.log(train_loss=train_loss,
                                    pi_metrics_train=pi_metrics_train,
                                    val_loss=val_loss,
                                    pi_metrics_val=pi_metrics_val,
                                    val_values_lower=val_pred_lower,
                                    val_values_upper=val_pred_upper,
                                    step=step)
                if self.model_path is not None and step % 500 == 0:
                    save_model(model=model, directory=self.model_path)
                step += 1
        if self.model_path is not None:
            save_model(model=model, directory=self.model_path)
        return max_found
