
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
from utilities.io import save_model, read_model
from models.generator import Decoder, Discriminator
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
        self.__loss = nn.BCELoss()

    def __get_labels(self, value, batch_size):
        # Label smoothing (https://github.com/soumith/ganhacks):
        noise = 0.2*torch.randn((batch_size, 1), device=self.device)
        if value == 1:
            ones = torch.ones((batch_size, 1), dtype=torch.float, device=self.device)
            ones = torch.clip(ones + noise, 0, 1)
            return ones
        if value == 0:
            zeros = torch.zeros((batch_size, 1), dtype=torch.float).to(self.device)
            zeros = torch.clip(zeros + noise, 0, 1)
            return zeros
        raise ValueError("value should either be 0 or 1")

    def objective(self,
                  batch_size,
                  lr_generator,
                  lr_discriminator,
                  generator_num_hidden_layers,
                  discriminator_num_hidden_layers,
                  latent_dim,
                  pretrained_generator=None,
                  plot=False):

        print(locals())

        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        model_discriminator = Discriminator(num_layers=discriminator_num_hidden_layers,
                                            input_size=self.num_classes)

        if pretrained_generator is not None:
            model_generator = read_model(pretrained_generator).decoder
        else:
            model_generator = Decoder(num_hidden_layers=generator_num_hidden_layers,
                                      latent_dim=latent_dim,
                                      input_size=self.num_classes)
        model_discriminator.to(self.device)
        model_generator.to(self.device)

        wandb.watch(model_discriminator, log_freq=100)
        wandb.watch(model_generator, log_freq=100)

        optimizer_discriminator = optim.SGD(model_discriminator.parameters(), lr=lr_discriminator)
        optimizer_generator = optim.Adam(model_generator.parameters(), lr=lr_generator, betas=(0.5, 0.999))

        model_discriminator.train()  # NOTE: Very important! Otherwise we zero the gradient
        model_generator.train()  # NOTE: Very important! Otherwise we zero the gradient

        step = 0
        for iteration in range(self.iterations):
            for train_input in tqdm(dataloader):
                # model_discriminator.train()  # NOTE: Not needed if we dont call .eval()
                # model_generator.train()  # NOTE: Not needed if we dont call .eval()
                optimizer_discriminator.zero_grad()

                # Train with all real batch
                batch_size = train_input.size(0)
                output = model_discriminator(train_input)
                real_loss = self.__loss(input=output,
                                        target=self.__get_labels(value=1, batch_size=batch_size))

                # Train with all fake batch
                noise = torch.randn(batch_size, latent_dim, device=self.device, requires_grad=True)
                fake = model_generator(noise)

                output = model_discriminator(fake.detach())
                fake_loss = self.__loss(input=output,
                                        target=self.__get_labels(value=0, batch_size=batch_size))
                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                optimizer_discriminator.step()

                # Update generator network
                optimizer_generator.zero_grad()

                output = model_discriminator(fake)
                generator_loss = self.__loss(input=output,
                                             target=self.__get_labels(value=1, batch_size=batch_size))
                generator_loss.backward()
                optimizer_generator.step()

                # model_discriminator.eval()
                # model_generator.eval()
                # with torch.no_grad():
                #     val_pred, val_labels = model(self.val_dataset.inputs, origin="mixed")
                #     val_loss = self.__loss(pred=val_pred,
                #                            target=val_labels)
                    # l_vals.append(val_loss.item())
                    # max_found = max(max_found, -np.nanmean(l_vals))

                if plot and step % self.log_freq == 0:
                    self.logger.log(discriminator_loss=real_loss+fake_loss,
                                    generator_loss=generator_loss,
                                    val_loss=0,
                                    step=step)

                if self.model_path is not None and step % 500 == 0:
                    save_model(model=model_discriminator, directory=self.model_path + "_discriminator")
                    save_model(model=model_generator, directory=self.model_path + "_generator")
                step += 1
        if self.model_path is not None:
            save_model(model=model_discriminator, directory=self.model_path + "_discriminator")
            save_model(model=model_generator, directory=self.model_path + "_generator")
        
        # Return last mse and KL obtained in validation
        return None, None


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
                                        pretrained_generator=config["pretrained_generator"],
                                        plot=config["enable_logging"])

    return val_mse, val_KL
