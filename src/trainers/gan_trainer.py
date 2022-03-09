
import collections
import os
import sys

import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities.metrics import get_jensen_shannon, get_kl_divergence
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import save_model
from models.generator_gan import GAN
from loggers.gan_logger import GanLogger

class GanTrainer:
    def __init__(self,
                 iterations,
                 train_data,
                 val_data,
                 loging_path="../runs",
                 num_classes=72,
                 log_freq=100,
                 model_path=None,  # File where to save model learned weights None to not save
                 device=torch.device("cuda:0")):
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.device = device
        self.log_freq = log_freq
        self.model_path = model_path
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.logger = GanLogger()

    def __loss(self, pred, target):
        return nn.BCELoss()(pred, target)

    def objective(self,
                  batch_size,
                  lr_generator,
                  lr_discriminator,
                  generator_num_hidden_layers,
                  discriminator_num_hidden_layers,
                  latent_dim,
                  plot=False):

        print(locals())

        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        model = GAN(num_classes=self.num_classes,
                    generator_input_size=latent_dim,
                    generator_num_hidden_layers=generator_num_hidden_layers,
                    discriminator_num_hidden_layers=discriminator_num_hidden_layers,
                    device=self.device)
        model.to(self.device)

        wandb.watch(model, log_freq=100)

        optimizer_discriminator = optim.Adam(model.generator.parameters(), lr=lr_discriminator, weight_decay=1e-5)
        optimizer_generator = optim.Adam(model.discriminator.parameters(), lr=lr_generator, weight_decay=1e-5)
        # optimizer = optim.Adam([
            # {'params': model.generator.parameters(), 'lr': lr_generator},
            # {'params': model.discriminator.parameters(), 'lr': lr_discriminator}
        # ])

        # l_vals = collections.deque(maxlen=50)
        # max_found = -np.inf
        step = 0
        for iteration in range(self.iterations):
            for train_input in tqdm(dataloader):
                model.train()  # NOTE: Very important! Otherwise we zero the gradient
                model.discriminator.zero_grad()

                mixed_pred, mixed_labels = model(train_input, origin="mixed")
                mixed_loss = self.__loss(pred=mixed_pred,
                                         target=mixed_labels)
                mixed_loss.backward()

                # fake_pred, fake_labels = model(train_input, origin="fake")
                # fake_loss = self.__loss(pred=fake_pred,
                #                         target=fake_labels)
                # fake_loss.backward()

                optimizer_discriminator.step()

                model.generator.zero_grad()

                generator_pred, fake_labels = model(train_input, origin="fake")
                generator_loss = self.__loss(pred=generator_pred,
                                             target=1-fake_labels)
                generator_loss.backward()
                
                optimizer_generator.step()

                model.eval()
                with torch.no_grad():
                    val_pred, val_labels = model(self.val_dataset.inputs, origin="mixed")
                    val_loss = self.__loss(pred=val_pred,
                                           target=val_labels)
                    # l_vals.append(val_loss.item())
                    # max_found = max(max_found, -np.nanmean(l_vals))

                if plot and step % self.log_freq == 0:
                    self.logger.log(discriminator_loss=mixed_loss,
                                    generator_loss=generator_loss,
                                    val_loss=val_loss,
                                    step=step)

                if self.model_path is not None and step % 500 == 0:
                    save_model(model=model, directory=self.model_path)
                step += 1
        if self.model_path is not None:
            save_model(model=model, directory=self.model_path)
        
        # Return last mse and KL obtained in validation
        return None


def train_gan(config) -> float:
    """Train a classification model and get the validation score

    Args:
        config (dict): Including all the needed args
        to load data, and train the model 
    """
    from utilities.io import read_data_generator

    dev = "cuda" if config["device"] == "cuda" and torch.cuda.is_available(
    ) else "cpu"
    print("Using device:", dev)

    if config["enable_logging"]:
        wandb.init(project=config["wandb_project_id"],
                   entity='sig-net',
                   config=config,
                   name=config["model_id"])

    train_data, val_data = read_data_generator(device=dev,
                                               data_id=config['data_id'],
                                               cosmic_version=config['cosmic_version'])

    trainer = GanTrainer(iterations=config["iterations"],  # Passes through all dataset
                         train_data=train_data,
                         val_data=val_data,
                         num_classes=config["num_classes"],
                         device=torch.device(dev),
                         model_path=os.path.join(config["models_dir"], config["model_id"]))

    val_mse, val_KL = trainer.objective(batch_size=config["batch_size"],
                                        lr_generator=config["lr_generator"],
                                        lr_discriminator=config["lr_discriminator"],
                                        generator_num_hidden_layers=config["generator_num_hidden_layers"],
                                        discriminator_num_hidden_layers=config["discriminator_num_hidden_layers"],
                                        latent_dim=config["latent_dim"],
                                        plot=config["enable_logging"])

    return val_mse, val_KL
