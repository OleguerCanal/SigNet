import os

import numpy as np
from skopt.space import Real, Integer, Categorical
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from HyperParameterOptimizer.gaussian_process import GaussianProcessSearch
from model import SignatureNet
from utilities.train_dataset import TrainDataSet
from utilities.metrics import get_kl_divergence, get_cross_entropy2


class MetaParamOptimizer:
    def __init__(self, iterations, num_classes,
                 train_input, train_baseline, train_label,
                 val_input, val_baseline, val_label):
        self.iterations = iterations  # Now iteration refers to passes through all dataset
        self.num_classes = num_classes
        self.train_dataset = TrainDataSet(train_input=train_input,
                                          train_label=train_label,
                                          train_baseline=train_baseline)
        self.val_input = val_input
        self.val_baseline = val_baseline
        # Not sure why we have to do this
        # (claudia): because in the last column we save the number of mutations and we don't need them here. 
        self.val_label = val_label[:, :self.num_classes]

    def objective(self, batch_size, lr, num_neurons, num_hidden_layers):
        print(batch_size, lr, num_neurons, num_hidden_layers)
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        sn = SignatureNet(num_classes=self.num_classes,
                          num_hidden_layers=int(num_hidden_layers),
                          num_units=int(num_neurons))
        sn.to(torch.device("cuda:0"))
        optimizer = optim.Adam(sn.parameters(),
                               lr=lr)
        #writer = SummaryWriter(log_dir=os.path.join("test", "1"))
        l_val = []
        for iteration in range(self.iterations):
            for train_input, train_label, train_baseline in tqdm(dataloader):
                #k += 1
                # Not sure why this line
                train_label = train_label[:, :self.num_classes]

                optimizer.zero_grad()
                train_prediction = sn(train_input, train_baseline)
                l = get_kl_divergence(train_prediction, train_label)
                #writer.add_scalar(f'loss/KL-divergence', l, iteration)
                l.backward()
                optimizer.step()

                with torch.no_grad():
                    val_prediction = sn(self.val_input, self.val_baseline)
                    l_val.append(get_kl_divergence(
                        val_prediction, self.val_label).item())    

        # Negative because we have a maximizer
        result = -np.nanmean(l_val[-100:])
        result = result if not np.isnan(result) else float(-5.) # Set a high number if is nan
        print(result)
        del dataloader
        del sn
        del optimizer
        return result


if __name__ == "__main__":
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        print("GPU")
    else:  
        dev = "cpu"  
    device = torch.device(dev)  


    iterations = 5
    num_classes = 72

    import pandas as pd
    train_input = torch.tensor(pd.read_csv(
        "data/train_input_w01.csv", header=None).values, dtype=torch.float)
    train_input = train_input.to(device)
    train_baseline = torch.tensor(pd.read_csv(
        "data/train_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    train_baseline = train_baseline.to(device)
    train_label = torch.tensor(pd.read_csv(
        "data/train_label_w01.csv", header=None).values, dtype=torch.float)
    train_label = train_label.to(device)

    val_input = torch.tensor(pd.read_csv(
        "data/validation_input_w01.csv", header=None).values, dtype=torch.float)
    val_input = val_input.to(device)
    val_baseline = torch.tensor(pd.read_csv(
        "data/validation_w01_baseline_JS.csv", header=None).values, dtype=torch.float)
    val_baseline = val_baseline.to(device)
    val_label = torch.tensor(pd.read_csv(
        "data/validation_label_w01.csv", header=None).values, dtype=torch.float)
    val_label = val_label.to(device)

    mpo = MetaParamOptimizer(iterations=iterations,  # Passes through all dataset
                             num_classes=num_classes,
                             train_input=train_input,
                             train_baseline=train_baseline,
                             train_label=train_label,
                             val_input=val_input,
                             val_baseline=val_baseline,
                             val_label=val_label)

    batch_sizes = Integer(name='batch_size', low=10, high=1000)
    learning_rates = Real(name='lr', low=0.00001, high=0.01)
    neurons = Integer(name='num_neurons', low=20, high=1500)
    layers = Integer(name='num_hidden_layers', low=1, high=10)

    search_space = [batch_sizes, learning_rates, neurons, layers]
    fixed_space = {}

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=mpo.objective,
                                      input_file='bo_search.csv',  # Use None to start from zero
                                      output_file='bo_search_2.csv')  # Store tested points
    gp_search.init_session()
    best_metaparams, best_val = gp_search.get_maximum(
        n_calls=10,
        n_random_starts=0,
        noise=0.01,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)