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
from utilities.metrics import get_MSE
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

    def objective(self,
                  batch_size,
                  lr,
                  num_neurons_pos,
                  num_neurons_neg,
                  num_hidden_layers_pos,
                  num_hidden_layers_neg,
                  normalize_mut,
                  plot=False):
        print(batch_size, lr, num_neurons_pos, num_hidden_layers_pos,
              num_neurons_neg, num_hidden_layers_neg, normalize_mut)
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        model = ErrorFinder(num_classes=self.num_classes,
                            num_hidden_layers_pos=int(num_hidden_layers_pos),
                            num_units_pos=int(num_neurons_pos),
                            num_hidden_layers_neg=int(num_hidden_layers_neg),
                            num_units_neg=int(num_neurons_neg),
                            normalize_mut=int(normalize_mut))
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
                train_prediction_pos, train_prediction_neg = model(weights=train_weight_guess,
                                                                   num_mutations=num_mut)

                # Compute loss
                real_error = train_label - train_weight_guess
                real_error_pos, real_error_neg = self.__decompose_errors(
                    real_error.detach())
                train_loss = get_MSE(train_prediction_pos, real_error_pos) + \
                             get_MSE(train_prediction_neg, real_error_neg)
                train_loss.backward(retain_graph=True)
                optimizer.step()

                with torch.no_grad():
                    val_prediction_pos, val_prediction_neg = model(
                        self.val_weight_guess, self.val_num_mut)
                    val_real_error = self.val_label - self.val_weight_guess
                    val_real_error_pos, val_real_error_neg = self.__decompose_errors(
                        val_real_error.detach())
                    val_loss = get_MSE(val_prediction_pos, val_real_error_pos) +\
                        get_MSE(val_prediction_neg, val_real_error_neg)
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                if plot:
                    self.logger.log(train_loss=train_loss,
                                    val_loss=val_loss,
                                    step=step)
                if self.model_path is not None and step % 500 == 0:
                    torch.save(model.state_dict(), os.path.join(
                        self.model_path, self.experiment_id))
                step += 1
        return max_found
