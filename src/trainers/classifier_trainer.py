import collections
import copy
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import Classifier
from utilities.data_partitions import DataPartitions
from utilities.metrics import get_jensen_shannon, get_fp_fn_soft, get_classification_metrics
from loggers.classifier_logger import ClassifierLogger

class ClassifierTrainer:
    def __init__(self,
                 iterations,
                 train_data,
                 val_data,
                 sigmoid_params = [5000, 2000],
                 loging_path="../runs",
                 num_classes=72,
                 log_freq=100,
                 model_path=None,  # File where to save model learned weights None to not save
                 device=torch.device("cuda:0")):
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.sigmoid_params = sigmoid_params
        self.device = device
        self.log_freq = log_freq
        self.model_path = model_path
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.logger = ClassifierLogger()

    def __loss(self, prediction, label):
        return nn.BCELoss()(prediction, label)
         
    def objective(self,
                  batch_size,
                  lr,
                  num_hidden_layers,
                  num_units,
                  plot=False):

        print(batch_size, lr, num_hidden_layers, num_units)

        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        model = Classifier(num_hidden_layers=int(num_hidden_layers),
                          num_units=int(num_units),
                          sigmoid_params=self.sigmoid_params)
        model.to(self.device)

        # if plot:
        #     wandb.watch(model, log_freq=self.log_freq, log_graph=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        l_vals = collections.deque(maxlen=50)
        max_found = -np.inf
        step = 0
        for iteration in range(self.iterations):
            for train_input, train_label, prev_guess, num_mut in tqdm(dataloader):                  # MIRAR AIXO!!!!!!!!!!
                model.train()  # NOTE: Very important! Otherwise we zero the gradient
                optimizer.zero_grad()                
                train_prediction = model(train_input, num_mut)
                train_loss = self.__loss(prediction=train_prediction,
                                        label=train_label)

                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_prediction = model(
                        self.val_dataset.inputs, self.val_dataset.num_mut)
                    val_loss = self.__loss(prediction=val_prediction,
                                        label=self.val_dataset.labels)
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                if plot and step % self.log_freq == 0:
                    self.logger.log(train_loss=train_loss,
                                    train_prediction=train_prediction,
                                    train_label=train_label,
                                    val_loss=val_loss,
                                    val_prediction=val_prediction,
                                    val_label=self.val_dataset.labels,
                                    step=step)

                if self.model_path is not None and step % 500 == 0:
                    directory = os.path.dirname(self.model_path)
                    pathlib.Path(directory).mkdir(
                        parents=True, exist_ok=True)
                    torch.save(model.state_dict(), self.model_path)
                step += 1
        return max_found
