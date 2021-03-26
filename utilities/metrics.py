import torch

def get_cosine_similarity(predicted_batch, label_batch):
    predicted_batch = torch.nn.functional.normalize(predicted_batch, dim=1, p=2)
    label_batch = torch.nn.functional.normalize(label_batch, dim=1, p=2)
    return torch.mean(torch.abs(torch.einsum("ij,ij->i", (predicted_batch, label_batch))))

def get_entropy(predicted_batch):
    batch_size = predicted_batch.shape[0]
    entropy = 0
    for i in range(batch_size):
        entropy += torch.distributions.categorical.Categorical(probs=predicted_batch[i, :].detach()).entropy()
    return entropy/batch_size