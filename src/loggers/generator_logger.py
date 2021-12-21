import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.metrics import get_MSE, get_kl_divergence


class GeneratorLogger:
    def __init__(self):

        self.metrics = {
            "mse": get_MSE,
        }

    def log(self,
            train_loss,
            train_prediction,
            train_label,
            train_mu,
            train_sigma, 
            val_loss,
            val_prediction,
            val_label,
            val_mu,
            val_sigma,
            step):

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})

        for metric_name in self.metrics.keys():
            metric = self.metrics[metric_name]
            wandb.log({"train_" + metric_name: metric(train_prediction, train_label).item()})
            wandb.log({"val_" + metric_name: metric(val_prediction, val_label).item()})

        
        def KL(mu, sigma):
            return (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        
        wandb.log({"train_KL": KL(train_mu, train_sigma)})
        wandb.log({"val_KL": KL(val_mu, val_sigma)})