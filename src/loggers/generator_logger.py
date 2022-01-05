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

        val_mse = get_MSE(val_prediction, val_label).item()

        # Between-examples variance
        train_variance = torch.mean(torch.var(train_prediction, dim=0)).item()
        val_variance = torch.mean(torch.var(val_prediction, dim=0)).item()
        wandb.log({"train_pred_variance": train_variance})
        wandb.log({"val_pred_variance": val_variance})

        # Within-examples variance
        train_variance = torch.mean(torch.var(train_mu, dim=1)).item()
        val_variance = torch.mean(torch.var(val_mu, dim=1)).item()
        wandb.log({"train_mu_variance": train_variance})
        wandb.log({"val_mu_variance": val_variance})
        
        def KL(mu, sigma):
            return torch.mean((sigma**2 + mu**2)/2. - torch.log(sigma) - 1/2)
        
        val_KL = KL(val_mu, val_sigma).item()
        wandb.log({"train_KL": KL(train_mu, train_sigma).item()})
        wandb.log({"val_KL": val_KL})

        wandb.log({"train_mu": torch.mean(train_mu)})
        wandb.log({"val_mu": torch.mean(val_mu)})

        wandb.log({"train_sigma": torch.mean(train_sigma)})
        wandb.log({"val_sigma": torch.mean(val_sigma)})

        return val_mse, val_KL