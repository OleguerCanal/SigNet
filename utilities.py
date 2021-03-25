import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

def plot_signature(signature, labels):
    plt.bar(range(96), signature, tick_label=labels)
    plt.xticks(rotation=90)
    plt.show()

def get_data_batches(signatures, batch_size, n_samples=500):
    input_batch = torch.empty(batch_size, 96)
    label_batch = torch.empty(batch_size, dtype=torch.long)

    for i in range(batch_size):
        label = random.randint(0, len(signatures)-1)
        signature = signatures[label]
        c = torch.distributions.categorical.Categorical(probs=signature)
        samples = c.sample(sample_shape=torch.Size([n_samples,])).type(torch.float32)
        sample = torch.histc(samples, bins=96, min=0, max=95)/float(n_samples)
        # plot_signature(signature=signature.numpy(), labels=None)
        # plot_signature(signature=sample.numpy(), labels=None)
        input_batch[i, :] = sample
        # label_batch[i, :] = torch.nn.functional.one_hot(torch.tensor(label),  num_classes=72)
        label_batch[i] = torch.tensor(label).type(torch.long)
    return input_batch, label_batch

def get_entropy(predicted_batch):
    batch_size = predicted_batch.shape[0]
    entropy = 0
    for i in range(batch_size):
        entropy += torch.distributions.categorical.Categorical(probs=predicted_batch[i, :].detach()).entropy()
    return entropy/batch_size

def confusion_matrix_plot(label_list, predicted_list, class_names):
    conf_mat = confusion_matrix(label_list.numpy(), predicted_list.numpy())
    plt.figure(figsize=(15,10))

    df_cm = pd.DataFrame(conf_mat, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return conf_mat