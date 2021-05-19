import os

from torch.utils.tensorboard import SummaryWriter


class ErrorFinderLogger:
    def __init__(self, path, experiment_id):
        self.writer = SummaryWriter(
            log_dir=os.path.join("runs/train", experiment_id))
        self.val_writer = SummaryWriter(
            log_dir=os.path.join("runs/val", experiment_id))


    def log(self,
            train_loss,
            train_in_prop,
            train_pi_width,
            val_loss,
            val_in_prop,
            val_pi_width,
            val_values_lower,
            val_values_upper,
            step):

        self.writer.add_scalar("metrics/Loss", train_loss.item(), step)
        self.writer.add_scalar("metrics/in_prop", train_in_prop.item(), step)
        self.writer.add_scalar("metrics/pi_width", train_pi_width.item(), step)
        self.val_writer.add_scalar("metrics/Loss", val_loss.item(), step)
        self.val_writer.add_scalar("metrics/in_prop", val_in_prop.item(), step)
        self.val_writer.add_scalar("metrics/pi_width", val_pi_width.item(), step)

        self.val_writer.add_histogram("histograms/val_low", val_values_lower, step)
        self.val_writer.add_histogram("histograms/val_up", val_values_upper, step)
