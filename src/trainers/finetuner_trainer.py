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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loggers.finetuner_logger import FinetunerLogger
from utilities.metrics import get_kl_divergence, get_fp_fn, get_fp_fn_soft
from utilities.train_dataset import TrainDataSet
from models.finetuner import FineTuner

class FinetunerTrainer:
    def __init__(self,
                 iterations,
                 train_input,
                 train_weight_guess,
                 train_label,
                 val_input,
                 val_weight_guess,
                 val_label,
                 fp_param=1e-3,
                 fn_param=1e-3,
                 loging_path="../runs",
                 experiment_id="test",
                 num_classes=72,
                 model_path=None,  # Path where to save model learned weights None to not save
                 device=torch.device("cuda:0")):
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.fp_param = fp_param
        self.fn_param = fn_param
        self.device = device
        self.model_path = model_path
        self.experiment_id = experiment_id
        self.train_dataset = TrainDataSet(train_input=train_input,
                                          train_label=train_label,
                                          train_baseline=train_weight_guess)
        self.val_input = val_input
        self.val_weight_guess = val_weight_guess
        self.val_label = val_label[:, :self.num_classes]
        self.logger = FinetunerLogger(path=loging_path, experiment_id=experiment_id)

    def __loss(self, prediction, label, FP, FN):
        l = get_kl_divergence(prediction, label)
        l += self.fp_param*FP/prediction.shape[0] + self.fn_param*FN/prediction.shape[0]
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
                          num_units=int(num_units))
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        l_vals = collections.deque(maxlen=100)
        max_found = -np.inf
        step = 0
        for iteration in range(self.iterations):
            for train_input, train_label, train_weight_guess in tqdm(dataloader):
                train_label = train_label[:, :self.num_classes]

                optimizer.zero_grad()
                train_prediction = model(train_input, train_weight_guess)
                train_FP, train_FN = get_fp_fn_soft(label_batch=train_label,
                                            prediction_batch=train_prediction)
                train_loss = self.__loss(prediction=train_prediction,
                                label=train_label,
                                FP=train_FP,
                                FN=train_FN)
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    val_prediction = model(
                        self.val_input, self.val_weight_guess)
                    val_FP, val_FN = get_fp_fn_soft(label_batch=self.val_label,
                                            prediction_batch=val_prediction)
                    val_loss = self.__loss(prediction=val_prediction,
                                           label=self.val_label,
                                           FP=val_FP,
                                           FN=val_FN)
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                if plot:
                    self.logger.log(train_loss=train_loss,
                                    train_prediction=train_prediction,
                                    train_label=train_label,
                                    train_FP=train_FP,
                                    train_FN=train_FN,
                                    val_loss=val_loss,
                                    val_prediction=val_prediction,
                                    val_label=self.val_label,
                                    val_FP=val_FP,
                                    val_FN=val_FN,
                                    step=step)

                if self.model_path is not None and step % 500 == 0:
                    pathlib.Path(self.model_path).mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(self.model_path, self.experiment_id))
                step += 1
        return max_found