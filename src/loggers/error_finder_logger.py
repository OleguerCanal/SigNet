import os

from torch.utils.tensorboard import SummaryWriter
import wandb

class ErrorFinderLogger:
    def __init__(self, path, experiment_id):
        # self.writer = SummaryWriter(
        #     log_dir=os.path.join(path, "train", experiment_id))
        # self.val_writer = SummaryWriter(
        #     log_dir=os.path.join(path, "val", experiment_id))
        pass


    def log(self,
            train_loss,
            pi_metrics_train,
            val_loss,
            pi_metrics_val,
            val_values_lower,
            val_values_upper,
            step):

        # self.writer.add_scalar("metrics/Loss", train_loss.item(), step)
        # self.val_writer.add_scalar("metrics/Loss", val_loss.item(), step)
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})

        # self.val_writer.add_histogram("histograms/val_low", val_values_lower, step)
        # self.val_writer.add_histogram("histograms/val_up", val_values_upper, step)
        wandb.log({"val_low": val_values_lower.cpu().detach().numpy()})
        wandb.log({"val_upp": val_values_upper.cpu().detach().numpy()})

        for key in pi_metrics_train.keys():
            wandb.log({"train_%s"%key: pi_metrics_train[key].item()})
        
        for key in pi_metrics_val.keys():
            wandb.log({"val_%s"%key: pi_metrics_val[key].item()})

        # for key in pi_metrics_val.keys():
        #     self.writer.add_scalar("metrics/%s"%key, pi_metrics_train[key].item(), step)
        #     self.val_writer.add_scalar("metrics/%s"%key, pi_metrics_val[key].item(), step)