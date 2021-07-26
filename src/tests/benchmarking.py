import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


def compute_mean_distance_by_sig(label_batch, guessed_batch):
    error_batch = torch.abs(label_batch - guessed_batch)
    mean_error = torch.mean(error_batch, dim=0)
    return mean_error


def compute_mean_distance_by_sig_interval(label_batch, pred_upper, pred_lower):
    lower = label_batch - pred_lower
    upper = pred_upper - label_batch
    lower = nn.ReLU()(-lower)
    upper = nn.ReLU()(-upper)
    distance = lower + upper
    mean_distance = torch.mean(distance, dim=0)
    return mean_distance


def plot_distance(label_batch, list_of_paths, list_of_names):
    # list_of_paths to the guessed_batch of the different algorithms
    for i in range(len(list_of_paths)):
        guessed_batch = torch.tensor(pd.read_csv(list_of_paths[i], header=None).values, dtype=torch.float)
        dist_method = compute_mean_distance_by_sig(label_batch, guessed_batch)
        plt.plot(dist_method, label=list_of_names[i])
    plt.legend()
    data = pd.read_excel("../../data/data.xlsx")
    plt.xticks(range(72), list(data.columns)[2:], rotation='vertical')
    plt.show()
    plt.close()

def plot_distance_interval(label_batch, list_of_paths, list_of_names, pred_upper, pred_lower):
    # list_of_paths to the guessed_batch of the different algorithms
    for i in range(len(list_of_paths)):
        guessed_batch = torch.tensor(pd.read_csv(list_of_paths[i], header=None).values, dtype=torch.float)
        dist_method = compute_mean_distance_by_sig(label_batch, guessed_batch)
        plt.plot(dist_method, label=list_of_names[i])
    dist_signatures_net = compute_mean_distance_by_sig_interval(label_batch, pred_upper, pred_lower)
    plt.plot(dist_signatures_net, label="signatures-net intervals")
    plt.legend()
    data = pd.read_excel("../../data/data.xlsx")
    plt.xticks(range(72), list(data.columns)[2:], rotation='vertical')
    plt.show()
    plt.close()


# Everything with realistic dataset
num_classes = 72
label_mut_batch = torch.tensor(pd.read_csv(
        "../../data/realistic_data/realistic_test_label.csv", header=None).values, dtype=torch.float)
label_batch = label_mut_batch[:, :num_classes]

pred_upper = torch.tensor(pd.read_csv(
        "../../data/realistic_data/methods/signatures-net_realistic_test_pos_guess.csv", header=None).values, dtype=torch.float)
perd_lower = torch.tensor(pd.read_csv(
        "../../data/realistic_data/methods/signatures-net_realistic_test_neg_guess.csv", header=None).values, dtype=torch.float)

list_of_paths = ["../../data/realistic_data/methods/decompTumor2Sig_realistic_test_guess.csv", "../../data/realistic_data/methods/mutSignatures_realistic_test_guess.csv", 
    "../../data/realistic_data/methods/SignatureEstimationQP_realistic_test_guess.csv", "../../data/realistic_data/methods/MutationalPatterns_realistic_test_guess.csv", 
    "../../data/realistic_data/methods/YAPSA_realistic_test_guess.csv", "../../data/realistic_data/methods/deconstructSigs_realistic_test_guess.csv",
    "../../data/realistic_data/methods/signatures-net_realistic_test_guess.csv"]
list_of_names = ["decompTumor2Sig", "mutSignatures", "SignatureEstimationQP", "MutationalPatterns", "YAPSA", "deconstructSigs", "signatures-net"]
        
plot_distance(label_batch, list_of_paths, list_of_names)
plot_distance_interval(label_batch, list_of_paths, list_of_names, pred_upper, perd_lower)


# Everything with random dataset
num_classes = 72
label_mut_batch = torch.tensor(pd.read_csv(
        "../../data/random_data/test_label_w01.csv", header=None).values, dtype=torch.float)
label_batch = label_mut_batch[:, :num_classes]

pred_upper = torch.tensor(pd.read_csv(
        "../../data/random_data/methods/signatures-net_random_test_pos_guess.csv", header=None).values, dtype=torch.float)
perd_lower = torch.tensor(pd.read_csv(
        "../../data/random_data/methods/signatures-net_random_test_neg_guess.csv", header=None).values, dtype=torch.float)

list_of_paths = ["../../data/random_data/methods/decompTumor2Sig_random_test_guess.csv", "../../data/random_data/methods/mutSignatures_random_test_guess.csv", 
    "../../data/random_data/methods/SignatureEstimationQP_random_test_guess.csv", "../../data/random_data/methods/MutationalPatterns_random_test_guess.csv", 
    "../../data/random_data/methods/YAPSA_random_test_guess.csv", "../../data/random_data/methods/deconstructSigs_random_test_guess.csv",
    "../../data/random_data/methods/signatures-net_random_test_guess.csv"]
list_of_names = ["decompTumor2Sig", "mutSignatures", "SignatureEstimationQP", "MutationalPatterns", "YAPSA", "deconstructSigs", "signatures-net"]
        
plot_distance(label_batch, list_of_paths, list_of_names)
plot_distance_interval(label_batch, list_of_paths, list_of_names, pred_upper, perd_lower)