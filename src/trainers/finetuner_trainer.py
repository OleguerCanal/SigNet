import collections
import copy
import os
import pathlib
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.finetuner import FineTuner
from utilities.data_partitions import DataPartitions
from utilities.metrics import get_jensen_shannon, get_fp_fn_soft, get_classification_metrics
from loggers.finetuner_logger import FinetunerLogger

class FinetunerTrainer:
    def __init__(self,
                 iterations,
                 train_data,
                 val_data,
                 fp_param=1e-3,
                 fn_param=1e-3,
                 sigmoid_params = [5000, 2000],
                 loging_path="../runs",
                 num_classes=72,
                 model_path=None,  # File where to save model learned weights None to not save
                 device=torch.device("cuda:0")):
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.fp_param = fp_param
        self.fn_param = fn_param
        self.sigmoid_params = sigmoid_params
        self.device = device
        self.model_path = model_path
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.logger = FinetunerLogger(
            path=loging_path,
            experiment_id="_".join(model_path.split("/")[-2:]))

    def __loss(self, prediction, label, FP, FN):
        l = get_jensen_shannon(prediction, label)
        l += self.fp_param*FP / \
            prediction.shape[0] + self.fn_param*FN/prediction.shape[0]
        return l

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
        model = FineTuner(num_classes=self.num_classes,
                          num_hidden_layers=int(num_hidden_layers),
                          num_units=int(num_units),
                          sigmoid_params=self.sigmoid_params)
        model.to(self.device)

        log_freq = 5
        if plot:
            wandb.watch(model, log_freq=log_freq, log_graph=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        l_vals = collections.deque(maxlen=100)
        max_found = -np.inf
        step = 0
        for iteration in range(self.iterations):
            for train_input, train_label, train_weight_guess, num_mut in tqdm(dataloader):
                model.train()  # NOTE: Very important! Otherwise we zero the gradient
                optimizer.zero_grad()                
                train_prediction = model(train_input, train_weight_guess, num_mut)
                train_FP, train_FN = get_fp_fn_soft(label_batch=train_label,
                                                    prediction_batch=train_prediction)
                train_loss = self.__loss(prediction=train_prediction,
                                        label=train_label,
                                        FP=train_FP,
                                        FN=train_FN)

                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_classification_metrics = get_classification_metrics(label_batch=train_label,
                                                                            prediction_batch=train_prediction)
                    val_prediction = model(
                        self.val_dataset.inputs, self.val_dataset.prev_guess, self.val_dataset.num_mut)
                    val_FP, val_FN = get_fp_fn_soft(label_batch=self.val_dataset.labels,
                                                    prediction_batch=val_prediction)
                    val_loss = self.__loss(prediction=val_prediction,
                                        label=self.val_dataset.labels,
                                        FP=val_FP,
                                        FN=val_FN)
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                    val_classification_metrics = get_classification_metrics(label_batch=self.val_dataset.labels,
                                                                            prediction_batch=val_prediction)

                if plot and step % log_freq == 0:
                    self.logger.log(train_loss=train_loss,
                                    train_prediction=train_prediction,
                                    train_label=train_label,
                                    train_classification_metrics=train_classification_metrics,
                                    val_loss=val_loss,
                                    val_prediction=val_prediction,
                                    val_label=self.val_dataset.labels,
                                    val_classification_metrics=val_classification_metrics,
                                    step=step)

                if self.model_path is not None and step % 500 == 0:
                    directory = os.path.dirname(self.model_path)
                    pathlib.Path(directory).mkdir(
                        parents=True, exist_ok=True)
                    torch.save(model.state_dict(), self.model_path)
                step += 1
        return max_found
