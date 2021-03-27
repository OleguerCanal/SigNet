import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

def plot_signature(signature, labels):
    plt.bar(range(96), signature, tick_label=labels)
    plt.xticks(rotation=90)
    plt.show()


def plot_confusion_matrix(label_list, predicted_list, class_names):
    conf_mat = confusion_matrix(label_list.numpy(), predicted_list.numpy())
    plt.figure(figsize=(15, 10))

    df_cm = pd.DataFrame(conf_mat, index=class_names,
                         columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return conf_mat


def probs_batch_to_sigs(label_batch, predicted_batch, cutoff, num_classes):
    label_sigs_list = torch.zeros(0, dtype=torch.long)
    predicted_sigs_list = torch.zeros(0, dtype=torch.long)
    for i in range(len(label_batch)):
        for j in range(len(label_batch[i])):
            if label_batch[i][j] > cutoff and predicted_batch[i][j] > cutoff:
                label_sigs_list = torch.cat([label_sigs_list, torch.from_numpy(np.array([j]))])
                predicted_sigs_list = torch.cat([predicted_sigs_list, torch.from_numpy(np.array([j]))])
            if label_batch[i][j] > cutoff and predicted_batch[i][j] < cutoff:  
                label_sigs_list = torch.cat([label_sigs_list, torch.from_numpy(np.array([j]))])
                predicted_sigs_list = torch.cat([predicted_sigs_list, torch.from_numpy(np.array([num_classes]))])
            if label_batch[i][j] < cutoff and predicted_batch[i][j] > cutoff:  
                label_sigs_list = torch.cat([label_sigs_list, torch.from_numpy(np.array([num_classes]))])
                predicted_sigs_list = torch.cat([predicted_sigs_list, torch.from_numpy(np.array([j]))])
               
    return label_sigs_list, predicted_sigs_list
