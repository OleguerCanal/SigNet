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
            val_loss,
            step):

        self.writer.add_scalar("metrics/Loss", train_loss.item(), step)
        self.val_writer.add_scalar("metrics/Loss", val_loss.item(), step)
