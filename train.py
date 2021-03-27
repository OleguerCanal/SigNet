import os

import pandas as pd
from snorkel import classification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SignatureNet
from utilities.dataloader import DataLoader
from utilities.metrics import get_cosine_similarity, get_entropy, get_divergence
from utilities.plotting import plot_signature, plot_confusion_matrix, probs_batch_to_sigs

# Model params
num_hidden_layers = 4
num_neurons = 500
num_classes = 10
intial_learning_rate = 0.01
learning_rate_steps = 20000
learning_rate_gamma = 0.1

# Training params
experiment_id = "test_4"
iterations = 1e3
batch_size = 50
num_samples = 5000

if __name__ == "__main__":
    data = pd.read_excel("data.xlsx")
    signatures = [torch.tensor(data.iloc[:, i]).type(
        torch.float32) for i in range(2, 74)][:num_classes]

    data_loader = DataLoader(signatures=signatures,
                             batch_size=batch_size,
                             n_samples=num_samples,
                             min_n_signatures=1,
                             max_n_signatures=5)

    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_id))

    sn = SignatureNet(num_classes=num_classes,
                      num_hidden_layers=num_hidden_layers, num_units=num_neurons)
    optimizer = optim.Adam(sn.parameters(), lr=intial_learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=learning_rate_steps, gamma=learning_rate_gamma)
    #loss = nn.CrossEntropyLoss()

    predicted_list = torch.zeros(0, dtype=torch.long)
    label_list = torch.zeros(0, dtype=torch.long)

    kl_div = torch.nn.KLDivLoss(reduction = 'batchmean')
    for iteration in tqdm(range(int(iterations))):
        input_batch, label_batch = data_loader.get_batch()
        optimizer.zero_grad()

        predicted_batch = sn(input_batch)

        if iteration > iterations/1.5:
            label_sigs, predicted_sigs = probs_batch_to_sigs(label_batch, predicted_batch, 0.1, num_classes)
            label_list = torch.cat([label_list, label_sigs.view(-1)])
            predicted_list = torch.cat([predicted_list, predicted_sigs.view(-1)])
        
        l = classification.cross_entropy_with_probs(predicted_batch, label_batch)

        writer.add_scalar(f'metrics/loss', l.item(), iteration)
        writer.add_scalar(f'metrics/cosine_similarity', get_cosine_similarity(predicted_batch, label_batch), iteration)
        writer.add_scalar(f'metrics/loss-entropy', l.item()-get_divergence(label_batch), iteration)
        writer.add_scalar(f'metrics/KL-divergence', kl_div(label_batch.log_softmax(0), predicted_batch.log_softmax(0)), iteration)
        l.backward()
        optimizer.step()
        scheduler.step()

    torch.save(sn.state_dict(), os.path.join("models", experiment_id))
    conf_mat = plot_confusion_matrix(label_list, predicted_list, range(num_classes+1))

    sm = torch.nn.Softmax()
    # for i in range(num_classes):
    #    prediction = sn(signatures[i].unsqueeze(dim=0))
    #    probabilities = sm(prediction)
    #    print(i)
    #    print(probabilities)

    prediction = sn((signatures[1]*0.3 + signatures[2]*0.7).unsqueeze(dim=0))
    probabilities = sm(prediction)
    print(probabilities)
