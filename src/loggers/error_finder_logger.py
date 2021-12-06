import os

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

class ErrorFinderLogger:
    def __init__(self, path, experiment_id):
        # self.writer = SummaryWriter(
        #     log_dir=os.path.join(path, "train", experiment_id))
        # self.val_writer = SummaryWriter(
        #     log_dir=os.path.join(path, "val", experiment_id))
        pass


    def _group_by_nummut(self, values, num_muts):
        """ Group by nummut and 
        """
        # num_muts = ((num_muts**0.6)/20.).to(torch.int)  # should split nummut into ~10 groups
        num_muts = torch.log10(num_muts).to(torch.int)  # should split nummut into ~10 groups
        unique_muts = num_muts.unique()
        # print(unique_muts)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # # vals = val_nummut.detach().cpu().numpy()
        # # plt.hist(np.log(vals), bins=1000)
        # plt.hist(num_muts.detach().cpu().numpy(), bins=100)
        # plt.show()
        # a = input()

        grouped_values = torch.empty_like(unique_muts, dtype=torch.float)
        for i, num_mut in enumerate(unique_muts):
            indices = num_muts.squeeze(1) == num_mut
            grouped_values[i] = torch.mean(values[indices, ...])
        return grouped_values, unique_muts

    def log(self,
            train_loss,
            pi_metrics_train,
            val_loss,
            pi_metrics_val,
            val_values_lower,
            val_values_upper,
            val_nummut,
            step):

        # self.writer.add_scalar("metrics/Loss", train_loss.item(), step)
        # self.val_writer.add_scalar("metrics/Loss", val_loss.item(), step)
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})

        # self.val_writer.add_histogram("histograms/val_low", val_values_lower, step)
        # self.val_writer.add_histogram("histograms/val_up", val_values_upper, step)
        # wandb.log({"val_low": val_values_lower.cpu().detach().numpy()})
        # wandb.log({"val_upp": val_values_upper.cpu().detach().numpy()})

        for key in pi_metrics_train.keys():
            wandb.log({"train_%s"%key: pi_metrics_train[key].item()})
        
        for key in pi_metrics_val.keys():
            if key == "in_prop" or key == "mean_interval_width":
                grouped, ranges = self._group_by_nummut(pi_metrics_val[key], val_nummut)
                for i, val in enumerate(grouped):
                    wandb.log({"val_%s_%i"%(key, i): val.item()})
            else:
                wandb.log({"val_%s"%key: torch.mean(pi_metrics_val[key]).item()})

        # for key in pi_metrics_val.keys():
        #     self.writer.add_scalar("metrics/%s"%key, pi_metrics_train[key].item(), step)
        #     self.val_writer.add_scalar("metrics/%s"%key, pi_metrics_val[key].item(), step)