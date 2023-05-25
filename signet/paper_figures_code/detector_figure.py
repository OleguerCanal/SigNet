import pandas as pd
import torch


from signet.utilities.plotting import plot_metric_vs_mutations_classifier_superlow



ex_res = pd.read_excel('../../data/superlow/superlow_results.xlsx')
guess = ex_res['guess']
label = ex_res['label']
num_muts = ex_res['num_mut']

num_muts_list = list(num_muts)

guess[guess<0.5] = 0
guess[guess>=0.5] = 1
guess = torch.tensor(guess.values, dtype=torch.int64, device='cpu')
label = torch.tensor(label.values, dtype=torch.int64, device='cpu')

plot_metric_vs_mutations_classifier_superlow(guess, label, num_muts_list, plot_path = '../../plots/paper/final/')