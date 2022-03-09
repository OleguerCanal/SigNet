import os
import sys

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GanLogger:
    def __init__(self):
        pass

    def log(self,
            discriminator_loss,
            generator_loss,
            val_loss,
            step):

        wandb.log({"discriminator_loss": discriminator_loss})
        wandb.log({"generator_loss": generator_loss})
        wandb.log({"val_loss": val_loss})