import os
import pathlib
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.generator import Generator
from utilities.metrics import get_jensen_shannon
from loggers.generator_logger import GeneratorLogger

class GeneratorTrainer:

    def __init__(self, signatures, baseline, finetuner, train_data, iterations, model_path=None):
        self.__generator = Generator(baseline=baseline,
                                     finetuner=finetuner,
                                     signatures=signatures)
        self.train_dataset = train_data
        # self.val_dataset = val_data
        self.logger = GeneratorLogger()
        self.__iterations = iterations
        self.model_path = model_path

    def __loss(self, prediction, label):
        loss = get_jensen_shannon(predicted_label=prediction, true_label=label)
        return loss

    def objective(self, batch_size, lr, plot=False):
        model = self.__generator
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        optimizer = optim.Adam(
            model.parameters(), lr=lr)
        log_freq = 5

        for step in range(self.__iterations):
            for train_input, _, _, num_mut in tqdm(dataloader):
                model.train()  # NOTE: Very important! Otherwise we zero the gradient
                optimizer.zero_grad()
                train_prediction = model(train_input, num_mut)

                train_loss = self.__loss(prediction=train_prediction,
                                         label=train_input)

                train_loss.backward()
                optimizer.step()

                model.eval()
                # with torch.no_grad():
                #     val_prediction = model(
                #         self.val_dataset.inputs, self.val_dataset.num_mut)

                #     val_loss = self.__loss(prediction=val_prediction,
                #                            label=self.val_dataset.inputs)

                if plot and step % log_freq == 0:
                    self.logger.log(train_loss=train_loss,
                                    train_prediction=train_prediction,
                                    train_label=train_input,
                                    # val_loss=val_loss,
                                    # val_prediction=val_prediction,
                                    # val_label=self.val_dataset.inputs,
                                    step=step)

                if self.model_path is not None and step % 500 == 0:
                    directory = os.path.dirname(self.model_path)
                    pathlib.Path(directory).mkdir(
                        parents=True, exist_ok=True)
                    torch.save(model.state_dict(), self.model_path)
