
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
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.io import save_model
from utilities.generator_data import GeneratorData
from models.generator import Generator
from loggers.generator_logger import GeneratorLogger

class GeneratorTrainer:
    def __init__(self,
                 iterations,
                 train_data,
                 val_data,
                 lagrange_param=1.0,
                 loging_path="../runs",
                 num_classes=72,
                 log_freq=100,
                 model_path=None,  # File where to save model learned weights None to not save
                 device=torch.device("cuda:0")):
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.device = device
        self.log_freq = log_freq
        self.lagrange_param = lagrange_param
        self.model_path = model_path
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.logger = GeneratorLogger()

    def __loss(self, input, pred, z_mu, z_var):
        kl_div = torch.mean((z_var**2 + z_mu**2)/2. - torch.log(z_var) - 1/2)
        mse = nn.MSELoss()(input, pred)
        return self.batch_size_factor*(mse + self.adapted_lagrange_param*kl_div)

    def objective(self,
                  batch_size,
                  lr_encoder,
                  lr_decoder,
                  num_hidden_layers,
                  latent_dim,
                  plot=False):

        print(batch_size, lr_encoder, lr_decoder,
              num_hidden_layers, latent_dim)

        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        model = Generator(input_size=int(self.num_classes),
                          num_hidden_layers=int(num_hidden_layers),
                          latent_dim=int(latent_dim),
                          device=self.device.type)
        model.to(self.device)

        wandb.watch(model, log_freq=100)

        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        optimizer = optim.Adam([
            {'params': model.encoder_layers.parameters(), 'lr': lr_encoder},
            {'params': model.decoder_layers.parameters(), 'lr': lr_decoder}
        ])

        l_vals = collections.deque(maxlen=50)
        max_found = -np.inf
        step = 0
        total_steps = self.iterations*len(self.train_dataset)
        # self.batch_size_factor = batch_size/len(self.train_dataset)
        self.batch_size_factor = 1.
        for iteration in range(self.iterations):
            for train_input in tqdm(dataloader):
                model.train()  # NOTE: Very important! Otherwise we zero the gradient
                optimizer.zero_grad()
                train_pred, train_mean, train_var = model(train_input)
                # self.adapted_lagrange_param = self.lagrange_param
                self.adapted_lagrange_param = self.lagrange_param * \
                    float(total_steps - step)/float(total_steps)
                train_loss = self.__loss(input=train_input,
                                         pred=train_pred,
                                         z_mu=train_mean,
                                         z_var=train_var)

                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_pred, val_mean, val_var = model(
                        self.val_dataset.inputs)
                    val_loss = self.__loss(input=self.val_dataset.inputs,
                                           pred=val_pred,
                                           z_mu=val_mean,
                                           z_var=val_var)
                    l_vals.append(val_loss.item())
                    max_found = max(max_found, -np.nanmean(l_vals))

                if plot and step % self.log_freq == 0:
                    val_mse, val_KL = self.logger.log(train_loss=train_loss,
                                                      train_prediction=train_pred,
                                                      train_label=train_input,
                                                      val_loss=val_loss,
                                                      val_prediction=val_pred,
                                                      val_label=self.val_dataset.inputs,
                                                      train_mu=train_mean,
                                                      train_sigma=train_var,
                                                      val_mu=val_mean,
                                                      val_sigma=val_var,
                                                      step=step)

                if self.model_path is not None and step % 500 == 0:
                    save_model(model=model, directory=self.model_path)
                step += 1
        if self.model_path is not None:
            save_model(model=model, directory=self.model_path)
            model_results = pd.DataFrame({"batch_size": [batch_size],
                                          "lr_encoder": [lr_encoder],
                                          "lr_decoder": [lr_decoder],
                                          "num_hidden_layers": [num_hidden_layers],
                                          "latent_dim": [latent_dim],
                                          "lagrange_param": [self.lagrange_param],
                                          "adapted_lagrange_param": [self.adapted_lagrange_param],
                                          "batch_size_factor": [self.batch_size_factor],
                                          "val_mse": [val_mse],
                                          "val_KL": [val_KL],
                                          "val_loss": [val_loss.item()]})
            model_results.to_csv("../tmp/generator_models.csv", header=False, index=False, mode="a")
        return max_found


def train_generator(config) -> float:
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

    train_data, val_data = read_data_generator(device=dev)

    trainer = GeneratorTrainer(iterations=config["iterations"],  # Passes through all dataset
                               train_data=train_data,
                               val_data=val_data,
                               lagrange_param=config["lagrange_param"],
                               device=torch.device(dev),
                               model_path=os.path.join(config["models_dir"], config["model_id"]))

    min_val = trainer.objective(batch_size=config["batch_size"],
                                lr_encoder=config["lr_encoder"],
                                lr_decoder=config["lr_decoder"],
                                num_hidden_layers=config["num_hidden_layers"],
                                latent_dim=config["latent_dim"],
                                plot=config["enable_logging"])

    return min_val
