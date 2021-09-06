
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
from utilities.train_dataset import TrainDataSet
from utilities.metrics import distance_to_interval, get_pi_metrics
from models.error_finder import ErrorFinder
from loggers.error_finder_logger import ErrorFinderLogger

class ErrorTrainer:
    def __init__(self,
                 iterations,
                 train_input,
                 train_weight_guess,  # Guessed labels by finetuner
                 train_label,
                 val_input,
                 val_weight_guess,
                 val_label,
                 loging_path="../runs",
                 experiment_id="test",
                 num_classes=72,
                 model_path=None,
                 device="cuda:0"):
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.train_dataset = TrainDataSet(train_input=train_input,
                                          train_label=train_label,
                                          train_baseline=train_weight_guess)
        self.val_input = val_input
        self.val_weight_guess = val_weight_guess
        self.val_num_mut = torch.reshape(
            val_label[:, self.num_classes], (list(val_label.size())[0], 1))
        self.val_label = val_label[:, :self.num_classes]
        self.device = device
        self.model_path = model_path
        self.experiment_id = experiment_id
        self.logger = ErrorFinderLogger(
            path=loging_path, experiment_id=experiment_id)

    def __decompose_errors(self, real_error):
        real_error_pos = torch.einsum(
            "be,be->be", real_error, (real_error > 0).type(torch.int))
        real_error_neg = torch.einsum(
            "be,be->be", real_error, (real_error < 0).type(torch.int))
        return real_error_pos, real_error_neg

    def __loss(self, label, pred_lower, pred_upper, lagrange_mult=5e-2):
        batch_size = float(pred_lower.shape[0])
        lower = label - pred_lower
        upper = pred_upper - label
        lower = nn.ReLU()(-lower)
        upper = nn.ReLU()(-upper)
        # inverse_interval = nn.ReLU()(pred_lower - pred_upper)       # Penalize if the interval is inverted
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
            for train_input, train_label, train_weight_guess in tqdm(dataloader):
                num_mut = torch.reshape(
                    train_label[:, self.num_classes], (list(train_label.size())[0], 1))
                train_label = train_label[:, :self.num_classes]

                optimizer.zero_grad()
                train_pred_upper, train_pred_lower = model(weights=train_weight_guess,
                                                           num_mutations=num_mut)

                # Compute loss
                # train_loss = distance_to_interval(train_label, train_weight_guess, train_prediction_pos, train_prediction_neg, penalization=0.1)
                train_loss = self.__loss(label=train_label,
                                         pred_lower=train_pred_lower,
                                         pred_upper=train_pred_upper)
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    val_pred_upper, val_pred_lower = model(weights=self.val_weight_guess,
                                                           num_mutations=self.val_num_mut)
                    # val_loss = distance_to_interval(
                    #     self.val_label, self.val_weight_guess, val_prediction_pos, val_prediction_neg, penalization=0.1)
                    val_loss = self.__loss(label=self.val_label,
                                           pred_lower=val_pred_lower,
                                           pred_upper=val_pred_upper)
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                if plot:
                    pi_metrics_train = get_pi_metrics(train_label, train_pred_lower, train_pred_upper)
                    pi_metrics_val = get_pi_metrics(self.val_label, val_pred_lower, val_pred_upper)
                    self.logger.log(train_loss=train_loss,
                                    pi_metrics_train=pi_metrics_train,
                                    val_loss=val_loss,
                                    pi_metrics_val=pi_metrics_val,
                                    val_values_lower=val_pred_lower,
                                    val_values_upper=val_pred_upper,
                                    step=step)
                if self.model_path is not None and step % 500 == 0:
                    torch.save(model.state_dict(), os.path.join(
                        self.model_path, self.experiment_id))
                step += 1
        return max_found
