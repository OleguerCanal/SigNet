import os

from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from utilities.metrics import get_MSE,\
    get_negative_cosine_similarity,\
    get_cross_entropy2,\
    get_kl_divergence,\
    get_jensen_shannon,\
    get_wasserstein_distance



class ClassifierLogger:
    def __init__(self):
        self.metrics = {
            # "mse": get_MSE,
            # "cos": get_negative_cosine_similarity,
            # "cross_ent": get_cross_entropy2,
            # "KL": get_kl_divergence,
            # "JS": get_jensen_shannon,
            # "W": get_wasserstein_distance,
        }

    def accuracy(self, prediction, label):
        threshold = torch.tensor([0.5])
        prediction = (prediction>threshold).float()*1
        return torch.sum(prediction == label)/torch.numel(prediction)*100

    def log(self,
            train_loss,
            train_prediction,
            train_label,
            val_loss,
            val_prediction,
            val_label,
            step):

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})

        wandb.log({"train_accuracy": self.accuracy(train_prediction, train_label).item()})
        wandb.log({"val_accuracy": self.accuracy(val_prediction, val_label).item()})

