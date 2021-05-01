import torch
from scipy.stats import wasserstein_distance
from snorkel import classification
import copy
import numpy as np

def get_MSE(predicted_label, true_label):
    return torch.nn.MSELoss()(predicted_label, true_label)

def get_cosine_similarity(predicted_label, true_label):
    predicted_label = torch.nn.functional.normalize(
        predicted_label, dim=1, p=2)
    true_label = torch.nn.functional.normalize(true_label, dim=1, p=2)
    return torch.mean(torch.abs(torch.einsum("ij,ij->i", (predicted_label, true_label))))

def get_negative_cosine_similarity(predicted_label, true_label):
    return -get_cosine_similarity(predicted_label, true_label)

def get_entropy(predicted_label):
    batch_size = predicted_label.shape[0]
    entropy = torch.mean(torch.distributions.categorical.Categorical(
        probs=predicted_label.cpu().detach()).entropy())
    return entropy

def get_cross_entropy(predicted_label, true_label):
    return classification.cross_entropy_with_probs(predicted_label, true_label)

def get_cross_entropy2(predicted_label, true_label):
    #predicted_label_local = copy.deepcopy(predicted_label)
    #predicted_label_local += 1e-6
    return torch.mean(-torch.einsum("ij,ij->i",(true_label, torch.log(predicted_label))))


def get_kl_divergence(predicted_label, true_label):
    #predicted_label += 1e-6
    #true_label_local = copy.deepcopy(true_label)
    #true_label_local += 1e-6
    #predicted_label_local = copy.deepcopy(predicted_label)
    #predicted_label_local += 1e-6
    #true_label += 1e-6
    return get_cross_entropy2(predicted_label, true_label) - get_entropy(true_label)


def get_jensen_shannon(predicted_label, true_label):
    true_label_local = copy.deepcopy(true_label)
    true_label_local += 1e-6
    r = (predicted_label + true_label_local)/2
    term1 = torch.mean(torch.einsum("ij,ij->i",(predicted_label,torch.log(torch.div(predicted_label, r)))))
    term2 = torch.mean(torch.einsum("ij,ij->i",(true_label_local, torch.log(torch.div(true_label_local, r)))))
    return 0.5 * (term1 + term2) 


def get_wasserstein_distance(predicted_label, true_label):
    dist = 0
    for i in range(predicted_label.shape[0]):
        dist += wasserstein_distance(
            predicted_label[i, :].cpu().detach().numpy(), true_label[i, :].cpu().detach().numpy())
    return torch.from_numpy(np.array(dist/predicted_label.shape[0]))

def get_fp_fn(label_batch, prediction_batch, cutoff=0.05, num_classes=72):
    label_mask = (label_batch > cutoff).type(torch.int)
    prediction_mask = (prediction_batch > cutoff).type(torch.int)
    fp = torch.sum(label_mask - prediction_mask < 0)
    fn = torch.sum(label_mask - prediction_mask > 0)
    return fp, fn

def probs_batch_to_sigs(label_batch, prediction_batch, cutoff=0.05, num_classes=72):
    label_sigs_list = torch.zeros(0, dtype=torch.long)
    predicted_sigs_list = torch.zeros(0, dtype=torch.long)
    for i in range(len(label_batch)):
        for j in range(len(label_batch[i])):
            if label_batch[i][j] > cutoff and prediction_batch[i][j] > cutoff:
                label_sigs_list = torch.cat(
                    [label_sigs_list, torch.from_numpy(np.array([j]))])
                predicted_sigs_list = torch.cat(
                    [predicted_sigs_list, torch.from_numpy(np.array([j]))])
            if label_batch[i][j] > cutoff and prediction_batch[i][j] < cutoff:
                label_sigs_list = torch.cat(
                    [label_sigs_list, torch.from_numpy(np.array([j]))])
                predicted_sigs_list = torch.cat(
                    [predicted_sigs_list, torch.from_numpy(np.array([num_classes]))])
            if label_batch[i][j] < cutoff and prediction_batch[i][j] > cutoff:
                label_sigs_list = torch.cat(
                    [label_sigs_list, torch.from_numpy(np.array([num_classes]))])
                predicted_sigs_list = torch.cat(
                    [predicted_sigs_list, torch.from_numpy(np.array([j]))])
    return label_sigs_list, predicted_sigs_list

if __name__ == "__main__":
    torch.seed = 0
    a = torch.rand((4, 6))
    a = torch.nn.functional.normalize(a, dim=1, p=1)
    print(a)
    print(get_entropy(a))
    print(a)
    print(get_divergence(a))
