import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.metrics import *
from baseline import SignatureFinder
from model_tester import *

metrics = {
    "mse" : get_MSE,
    "cos" : get_negative_cosine_similarity,
    "cross_ent" : get_cross_entropy2,
    "KL" : get_kl_divergence,
    "JS" : get_jensen_shannon,
    "W" : get_wasserstein_distance,
}

metrics_interest = {
    "mse" : get_MSE,
    "JS" : get_jensen_shannon
}
if __name__=="__main__":
    # true_label = torch.tensor([0, 0, 0.3, 0, 0.2, 0.5, 0, 0], dtype=torch.double).unsqueeze(0)
    # prediction_1 = torch.tensor([0, 0, 0.4, 0, 0.15, 0.45, 0, 0], dtype=torch.double).unsqueeze(0)
    # prediction_2 = torch.tensor([0, 0.1, 0.3, 0, 0.15, 0.45, 0, 0], dtype=torch.double).unsqueeze(0)

    # for metric_name in list(metrics.keys()):
    #     a = metrics[metric_name](prediction_1, true_label).item()
    #     b = metrics[metric_name](prediction_2, true_label).item()
    #     print(metric_name, a, b)

    num_classes = 72
    data = pd.read_excel("../data/data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(torch.float32)
                  for i in range(2, 74)][:num_classes]
    sf = SignatureFinder(signatures)

    validation_input = torch.tensor(pd.read_csv("../data/validation_input.csv", header=None).values, dtype=torch.double)
    validation_label = torch.tensor(pd.read_csv("../data/validation_label.csv", header=None).values, dtype=torch.double)
    
    
    metrics_results = np.zeros((len(metrics), len(metrics)))
    N_total = 5000
   
    j = -1
    for optimizer_metric in list(metrics_interest.keys()):
        j += 1
        print(optimizer_metric)
        guessed_labels = torch.zeros(0, dtype=torch.long)
        for i in  tqdm(range(int(N_total))):
            sf.metric = metrics[optimizer_metric]
            pred = sf.get_weights(validation_input[i,:])
            pred = torch.tensor(pred, dtype=torch.double).unsqueeze(0)
            guessed_labels = torch.cat([guessed_labels, pred])
            #print(pred)
            # print(validation_label[i,:])
            k = -1
            for wight_metric in list(metrics.keys()):
                k += 1
                val = metrics[wight_metric](pred, validation_label[i,:].unsqueeze(0)).item()
                metrics_results[k,j] += val      
        model_tester = ModelTester(num_classes=num_classes)
        model_tester.test(guessed_labels=guessed_labels, true_labels=validation_label[:N_total,:])

    metrics_results = metrics_results/N_total
    metrics_results[1,:] = -metrics_results[1,:]
    df = pd.DataFrame(metrics_results)
    df.to_csv("../data/metrics_results.csv", header=False, index=False)       

    heatmap = sns.heatmap(df, annot=True, vmin=0, vmax=5)

    heatmap.yaxis.set_ticklabels(
        ['MSE', 'COS', 'CROSS_ENT', 'KL', 'JS', 'W'], rotation=0, ha='right', fontsize=6)
    heatmap.xaxis.set_ticklabels(
        ['MSE', 'COS', 'CROSS_ENT', 'KL', 'JS', 'W'], rotation=45, ha='right', fontsize=6)
    plt.ylabel('Weights metric', fontsize=8)
    plt.xlabel('Optimizer metric', fontsize=8)  
    plt.show() 

