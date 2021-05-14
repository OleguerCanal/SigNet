import copy

import numpy as np
from scipy.stats import wasserstein_distance
from snorkel import classification
import torch
import torch.nn as nn


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

def distance_to_interval(label_batch, train_weight_guess, prediction_pos, prediction_neg, penalization):
    lower_bound = train_weight_guess - abs(prediction_neg)
    lower = label_batch - lower_bound
    upper_bound = train_weight_guess + abs(prediction_pos)
    upper = prediction_pos - label_batch
    lower[lower>0] = 0
    upper[upper>0] = 0
    interval_length = upper_bound - lower_bound
    return torch.sum(abs(lower)) + torch.sum(abs(upper)) + penalization * torch.sum(interval_length)

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

def get_pi_metrics(label, pred_lower, pred_upper):
    """Used to compare prediction interval guesses.
    
    Returns:
        in_prop [float]: Proportion of labels in (pred_lower, pred_upper)
        mean_interval_width [float]: Mean width of the intervals
    """
    k_hu = (label <= pred_upper).type(torch.float)  # 1 if label < upper; else 0
    k_hl = (pred_lower <= label).type(torch.float)  # 1 if lower < label; else 0
    k_h = torch.einsum("be,be->be", k_hl, k_hu)  # 1 if label in (lower, upper) else 0
    in_prop = torch.mean(k_h)  # Hard Prediction Interval Coverage Probability
    mean_interval_width = torch.mean(torch.max(torch.zeros_like(label), pred_upper - pred_lower))
    return in_prop, mean_interval_width


def get_soft_qd_loss(label, pred_lower, pred_upper, conf=0.05, lagrange_mult=1e-4, softening_factor=100.0):
    """Used to optimize:
    
    min pred_upper - pred_lower
    s.t. p(label in (pred_lower, pred_upper)) >= 1 - conf

    Algorithm 1 of https://arxiv.org/pdf/1802.07167.pdf

    conf [float]: Interval confidence, proportion of missplaced points which is "ok" usually ~0.05
    lagrange_mult [float]: "How much do we care about missclassifications"
                            Larger lagrange_mult means more correctly guessed points at the expense of larger intervals
    softening_factor [float]: The bigger the closer it is to the real function
                              This means better guesses but harder optimization (less differentiable)
    """
    EPS_ = 1e-6

    # Hard in-between constrain
    k_hu = (label <= pred_upper).type(torch.float)  # 1 if label < upper; else 0
    k_hl = (pred_lower <= label).type(torch.float)  # 1 if lower < label; else 0
    k_h = torch.einsum("be,be->be", k_hl, k_hu)  # 1 if label in (lower, upper) else 0
    PICP_h = torch.mean(k_h)  # Prediction Interval Coverage Probability

    # Softened in-between constrain (same as before but differentiable)
    k_su = nn.Sigmoid()((pred_upper - label)*softening_factor)
    k_sl = nn.Sigmoid()((label - pred_lower)*softening_factor)
    k_s = torch.einsum("be,be->be", k_sl, k_su)
    PICP_s = torch.mean(k_s)  # Soft Prediction Interval Coverage Probability

    # Compute Mean Prediction Interval Width (MPIW)
    MPIW = torch.sum(torch.einsum("be,be->be", (pred_upper - pred_lower), k_h))/(torch.sum(k_h) + EPS_)

    # Compute constrain
    n = float(torch.numel(label))  # number elements in the input
    constrain = (n/(conf*(1.0 - conf)))*torch.max(torch.tensor(0).to(PICP_s), (1.0 - conf) - PICP_s)**2
    
    # Compute and return loss
    loss = MPIW + lagrange_mult*constrain
    return loss, PICP_h, MPIW

if __name__ == "__main__":
    torch.seed = 0

    u_s = [u/100. for u in range(100)]
    losses = []
    mpiws = []
    picps = []
    for u in u_s:
        lower = torch.tensor([[0.0]])
        label = torch.tensor([[0.25]])
        upper = torch.tensor([[u]])

        # in_prop, mean_interval_width = get_pi_metrics(label, lower, upper)
        # print("in_prop", in_prop)
        # print("mean_interval_width", mean_interval_width)

        loss, picp, mpiw = get_soft_qd_loss(label, lower, upper)
        # print("loss", loss)
        losses.append(loss)
        picps.append(picp)
        mpiws.append(mpiw)

    import matplotlib.pyplot as plt
    plt.plot(u_s, losses, label="loss")
    plt.plot(u_s, picps, label="in_prop")
    plt.plot(u_s, mpiws, label="width")
    plt.legend()
    plt.show()