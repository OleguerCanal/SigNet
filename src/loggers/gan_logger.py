import os
import sys

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GanLogger:
    def __init__(self):
        pass

    def log(self,
            train_loss,
            val_loss,
            step):

        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})