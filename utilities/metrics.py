import torch
from scipy.stats import wasserstein_distance
from snorkel import classification

def get_MSE(predicted_label, true_label):
    return torch.nn.MSELoss()(predicted_label, true_label)


def get_cosine_similarity(predicted_batch, label_batch):
    predicted_batch = torch.nn.functional.normalize(
        predicted_batch, dim=1, p=2)
    label_batch = torch.nn.functional.normalize(label_batch, dim=1, p=2)
    return torch.mean(torch.abs(torch.einsum("ij,ij->i", (predicted_batch, label_batch))))


def get_entropy(predicted_batch):
    batch_size = predicted_batch.shape[0]
    entropy = torch.mean(torch.distributions.categorical.Categorical(
        probs=predicted_batch.detach()).entropy())
    return entropy


def get_cross_entropy(predicted_label, true_label):
    return classification.cross_entropy_with_probs(predicted_label, true_label)

def get_cross_entropy2(predicted_label, true_label):
    return torch.mean(-torch.einsum("ij,ij->i",(true_label, torch.log(predicted_label))))


def get_kl_divergence(predicted_label, true_label):
    return get_cross_entropy2(predicted_label, true_label) - get_entropy(true_label)


def get_jensen_shannon(predicted_label, true_label):
    print(predicted_label)
    print(true_label)
    r = (predicted_label + true_label)/2
    term1 = torch.mean(torch.einsum("ij,ij->i",(predicted_label,torch.nan_to_num(torch.log(torch.div(predicted_label, r))))))
    term2 = torch.mean(torch.einsum("ij,ij->i",(true_label, torch.nan_to_num(torch.log(torch.div(true_label, r))))))
    return 0.5 * (term1 + term2) 


def get_wasserstein_distance(predicted_label, true_label):
    dist = 0
    for i in range(predicted_label.shape[0]):
        dist += wasserstein_distance(
            predicted_label[i, :].detach().numpy(), true_label[i, :].detach().numpy())
    return dist/predicted_label.shape[0]


if __name__ == "__main__":
    torch.seed = 0
    a = torch.rand((4, 6))
    a = torch.nn.functional.normalize(a, dim=1, p=1)
    print(a)
    print(get_entropy(a))
    print(a)
    print(get_divergence(a))
