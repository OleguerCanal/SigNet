
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


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.data_partitions import DataPartitions
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
            path=loging_path,
            experiment_id="_".join(model_path.split("/")[-2:]))

    def __loss(self, label, pred_lower, pred_upper, lagrange_mult=7e-3):
        batch_size = float(pred_lower.shape[0])
        lower = label - pred_lower
        upper = pred_upper - label
        lower = nn.ReLU()(-lower)
        upper = nn.ReLU()(-upper)
        # inverse_interval = nn.ReLU()(pred_lower - pred_upper)  # Penalize if the interval is inverted
        interval_length = ((pred_upper - pred_lower)**2)/batch_size
        loss_by_mutation_signature = interval_length + \
            lagrange_mult*(lower + upper)
        loss_by_mutation = torch.linalg.norm(
            1e4*loss_by_mutation_signature, ord=5, axis=1)
        loss = torch.mean(loss_by_mutation)
        return loss

    def objective(self,
                  batch_size,
                  lr,
                  num_neurons_pos,
                  num_neurons_neg,
                  num_hidden_layers_pos,
                  num_hidden_layers_neg,
                  plot=False):
        print(batch_size, lr, num_neurons_pos, num_hidden_layers_pos,
              num_neurons_neg, num_hidden_layers_neg)
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        model = ErrorFinder(num_classes=self.num_classes,
                            num_hidden_layers_pos=int(num_hidden_layers_pos),
                            num_units_pos=int(num_neurons_pos),
                            num_hidden_layers_neg=int(num_hidden_layers_neg),
                            num_units_neg=int(num_neurons_neg))
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(),
                               lr=lr)
        l_vals = collections.deque(maxlen=100)
        max_found = -np.inf
        step = 0
        for iteration in range(self.iterations):
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
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                if plot:
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
                    torch.save(model.state_dict(), self.model_path)
                step += 1
        return max_found
