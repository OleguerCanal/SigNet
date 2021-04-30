import os

import copy
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
from utilities.metrics import get_MSE


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
        self.val_num_mut = torch.reshape(val_label[:,self.num_classes], (list(val_label.size())[0],1))
        self.val_label = val_label[:, :self.num_classes]

    def objective(self, batch_size, lr, num_neurons_pos, num_neurons_neg, num_hidden_layers_pos,
                    num_hidden_layers_neg, normalize_mut):
        print(batch_size, lr, num_neurons_pos, num_hidden_layers_pos,  num_neurons_neg, num_hidden_layers_neg, normalize_mut)
        dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=int(batch_size),
                                shuffle=True)
        sn = SignatureNet(num_classes=self.num_classes,
                          num_hidden_layers_pos=int(num_hidden_layers_pos),
                          num_units_pos=int(num_neurons_pos),
                          num_hidden_layers_neg=int(num_hidden_layers_neg),
                          num_units_neg=int(num_neurons_neg),
                          normalize_mut=int(normalize_mut))
        sn.to(torch.device("cuda:0"))
        optimizer = optim.Adam(sn.parameters(),
                               lr=lr)
        #writer = SummaryWriter(log_dir=os.path.join("test", "1"))
        l_val = []
        for iteration in range(self.iterations):
            for train_input, train_label, train_baseline in tqdm(dataloader):
                #k += 1
                num_mut = torch.reshape(train_label[:,num_classes], (list(train_label.size())[0],1))
                train_label = train_label[:, :self.num_classes]

                optimizer.zero_grad()
                train_prediction_pos, train_prediction_neg = sn(train_baseline, num_mut)
                #l = get_MSE(train_prediction, abs(train_baseline - train_label))
                #writer.add_scalar(f'loss', l, iteration)
                real_error = train_label - train_baseline
                real_error_pos = copy.deepcopy(real_error)
                real_error_pos[real_error < 0] = 0
                real_error_neg = copy.deepcopy(real_error)
                real_error_neg[real_error > 0] = 0
                l = get_MSE(train_prediction_pos, real_error_pos) + get_MSE(train_prediction_neg, real_error_neg)
                l.backward()
                optimizer.step()

                with torch.no_grad():
                    val_prediction_pos, val_prediction_neg = sn(self.val_baseline, self.val_num_mut)
                    val_real_error = self.val_label - self.val_baseline
                    val_real_error_pos = copy.deepcopy(val_real_error)
                    val_real_error_pos[val_real_error < 0] = 0
                    val_real_error_neg = copy.deepcopy(val_real_error)
                    val_real_error_neg[val_real_error > 0] = 0
                    l_val.append(get_MSE(val_prediction_pos, val_real_error_pos).item()+get_MSE(val_prediction_neg, val_real_error_neg).item())    
                

        # Negative because we have a maximizer
        result = -np.nanmean(l_val[-100:])
        result = result if not np.isnan(result) else float(-5.) # Set a high number if is nan
        print(result)
        # del dataloader
        # del sn
        # del optimizer
        return result


if __name__ == "__main__":
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        print("GPU")
    else:  
        dev = "cpu"  
        print("CPU")
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

    batch_sizes = Integer(name='batch_size', low=50, high=1000)
    learning_rates = Real(name='lr', low=0.00001, high=0.005)
    neurons_pos = Integer(name='num_neurons_pos', low=20, high=1500)
    layers_pos = Integer(name='num_hidden_layers_pos', low=1, high=10)
    neurons_neg = Integer(name='num_neurons_neg', low=20, high=1500)
    layers_neg = Integer(name='num_hidden_layers_neg', low=1, high=10)
    normalize_mut_param = Integer(name='normalize_mut', low=1000, high=100000)

    search_space = [batch_sizes, learning_rates, neurons_pos, layers_pos, neurons_neg, layers_neg, normalize_mut_param]
    fixed_space = {}

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_space,
                                      evaluator=mpo.objective,
                                      input_file='search.csv',  # Use None to start from zero
                                      output_file='search2.csv')  # Store tested points
    gp_search.init_session()
    best_metaparams, best_val = gp_search.get_maximum(
        n_calls=100,
        n_random_starts=0,
        noise=0.01,
        verbose=True,
        plot_results=True)

    print("best_metaparams:", best_metaparams)
    print("best_val:", best_val)