import os

from torch.utils.tensorboard import SummaryWriter
import wandb

from utilities.metrics import get_MSE,\
    get_negative_cosine_similarity,\
    get_cross_entropy2,\
    get_kl_divergence,\
    get_jensen_shannon,\
    get_wasserstein_distance



class FinetunerLogger:
    def __init__(self, path, experiment_id):
        # self.writer = SummaryWriter(
        #     log_dir=os.path.join(path, "train", experiment_id))
        # self.val_writer = SummaryWriter(
        #     log_dir=os.path.join(path, "val", experiment_id))

        self.metrics = {
            # "mse": get_MSE,
            # "cos": get_negative_cosine_similarity,
            # "cross_ent": get_cross_entropy2,
            # "KL": get_kl_divergence,
            # "JS": get_jensen_shannon,
            # "W": get_wasserstein_distance,
        }

    def log(self,
            train_loss,
            train_prediction,
            train_label,
            train_classification_metrics,
            val_loss,
            val_prediction,
            val_label,
            val_classification_metrics,
            step):

        # self.writer.add_scalar("metrics/Loss", train_loss.item(), step)
        # self.val_writer.add_scalar("metrics/Loss", val_loss.item(), step)

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})

        for metric_name in self.metrics.keys():
            metric = self.metrics[metric_name]
            # self.writer.add_scalar("metrics/" + metric_name, metric(train_prediction, train_label).item(), step)
            # self.val_writer.add_scalar("metrics/" + metric_name, metric(val_prediction, val_label).item(), step)
            wandb.log({"train_" + metric_name: metric(train_prediction, train_label).item()})
            wandb.log({"val_" + metric_name: metric(val_prediction, val_label).item()})

        for metric_name in train_classification_metrics.keys():
            # self.writer.add_scalar("metrics/" + metric_name, train_classification_metrics[metric_name].item(), step)
            wandb.log({"train_" + metric_name: train_classification_metrics[metric_name].item()})

        for metric_name in val_classification_metrics.keys():
            # self.val_writer.add_scalar("metrics/" + metric_name, val_classification_metrics[metric_name].item(), step)
            wandb.log({"val_" + metric_name: val_classification_metrics[metric_name].item()})
