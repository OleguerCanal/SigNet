import os

from torch.utils.tensorboard import SummaryWriter

from utilities.metrics import get_MSE,\
    get_negative_cosine_similarity,\
    get_cross_entropy2,\
    get_kl_divergence,\
    get_jensen_shannon,\
    get_wasserstein_distance


class FinetunerLogger:
    def __init__(self, path, experiment_id):
        self.writer = SummaryWriter(
            log_dir=os.path.join("runs/train", experiment_id))
        self.val_writer = SummaryWriter(
            log_dir=os.path.join("runs/val", experiment_id))

        self.metrics = {
            "mse": get_MSE,
            "cos": get_negative_cosine_similarity,
            "cross_ent": get_cross_entropy2,
            "KL": get_kl_divergence,
            "JS": get_jensen_shannon,
            "W": get_wasserstein_distance,
        }

    def log(self,
            train_loss,
            train_prediction,
            train_label,
            train_FP,
            train_FN,
            val_loss,
            val_prediction,
            val_label,
            val_FP,
            val_FN,
            step):

        self.writer.add_scalar("metrics/Loss", train_loss.item(), step)
        self.val_writer.add_scalar("metrics/Loss", val_loss.item(), step)

        for metric_name in self.metrics.keys():
            metric = self.metrics[metric_name]
            self.writer.add_scalar("metrics/" + metric_name, metric(train_prediction, train_label).item(), step)
            self.val_writer.add_scalar("metrics/" + metric_name, metric(val_prediction, val_label).item(), step)

        self.writer.add_scalar("metrics/FP", train_FP.item(), step)
        self.val_writer.add_scalar("metrics/FP", val_FP.item(), step)

        self.writer.add_scalar("metrics/FN", train_FN.item(), step)
        self.val_writer.add_scalar("metrics/FN", val_FN.item(), step)
