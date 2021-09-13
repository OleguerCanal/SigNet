import os

from torch.utils.tensorboard import SummaryWriter
import wandb

from utilities.metrics import get_MSE,\
    get_negative_cosine_similarity,\
    get_cross_entropy2,\
    get_kl_divergence,\
    get_jensen_shannon,\
    get_wasserstein_distance


class GeneratorLogger:
    def __init__(self):

        self.metrics = {
            "mse": get_MSE,
            "cos": get_negative_cosine_similarity,
            # "cross_ent": get_cross_entropy2,
            # "KL": get_kl_divergence,
            # "JS": get_jensen_shannon,
            # "W": get_wasserstein_distance,
        }

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

        for metric_name in self.metrics.keys():
            metric = self.metrics[metric_name]
            wandb.log({"train_" + metric_name: metric(train_prediction, train_label).item()})
            wandb.log({"val_" + metric_name: metric(val_prediction, val_label).item()})