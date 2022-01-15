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
from utilities.io import save_model
from utilities.metrics import get_jensen_shannon, get_fp_fn_soft, get_classification_metrics, get_kl_divergence
from loggers.finetuner_logger import FinetunerLogger

class FinetunerTrainer:
    def __init__(self,
                 iterations,
                 train_data,
                 val_data,
                 network_type,
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
        self.network_type = network_type
        self.logger = FinetunerLogger()

    def __loss(self, prediction, label, FP, FN):
        assert (self.network_type in ['perturbed', 'realistic', 'generator'])
        if self.network_type == 'perturbed':
            fp_param = 1e-3
            fn_param = 0.1
            l = get_kl_divergence(prediction, label)
            l += fp_param*FP / prediction.shape[0] +\
                 fn_param*FN / prediction.shape[0]
        elif self.network_type == 'realistic' or self.network_type == 'generator':
            fp_param = 1e-3
            fn_param = 0.25
            l = get_kl_divergence(prediction, label)
            l += fp_param*FP / prediction.shape[0] +\
                 fn_param*FN / prediction.shape[0]
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

        # if plot:
        #     wandb.watch(model, log_freq=self.log_freq, log_graph=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        l_vals = collections.deque(maxlen=50)
        max_found = -np.inf
        step = 0
        for _ in range(self.iterations):
            for train_input, train_label, _, num_mut, _ in tqdm(dataloader):
                model.train()  # NOTE: Very important! Otherwise we zero the gradient
                optimizer.zero_grad()                
                train_prediction = model(train_input, num_mut)
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
                    val_prediction = model(self.val_dataset.inputs, self.val_dataset.num_mut)
                    val_FP, val_FN = get_fp_fn_soft(label_batch=self.val_dataset.labels,
                                                    prediction_batch=val_prediction)
                    val_loss = self.__loss(prediction=val_prediction,
                                           label=self.val_dataset.labels,
                                           FP=val_FP,
                                           FN=val_FN)
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                if plot and step % self.log_freq == 0:
                    val_classification_metrics = get_classification_metrics(label_batch=self.val_dataset.labels,
                                                                            prediction_batch=val_prediction)
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
                    save_model(model=model, directory=self.model_path)
                step += 1
        save_model(model=model, directory=self.model_path)
        return max_found

def train_finetuner(config) -> float:
    from utilities.io import read_data
    from models.finetuner import baseline_guess_to_finetuner_guess

    # Select training device
    dev = "cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    print("Using device:", device)

    # Set paths
    finetuner_path = os.path.join(config["models_dir"], config["model_id"])

    if config["enable_logging"]:
        wandb.init(project=config["wandb_project_id"],
                entity='sig-net',
                config=config,
                name=config["model_id"])

    # Load data
    train_data, val_data = read_data(experiment_id=config["data_id"],
                                     source=config["source"],
                                     device=dev)

    trainer = FinetunerTrainer(iterations=config["iterations"],  # Passes through all dataset
                               train_data=train_data,
                               val_data=val_data,
                               network_type=config["network_type"],
                               num_classes=config["num_classes"],
                               sigmoid_params=config["sigmoid_params"],
                               device=device,
                               model_path=finetuner_path)

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr=config["lr"],
                                num_hidden_layers=config["num_hidden_layers"],
                                num_units=config["num_neurons"],
                                plot=True)
    return min_val