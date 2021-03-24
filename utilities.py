import matplotlib.pyplot as plt
import random
import torch

def plot_signature(signature, labels):
    plt.bar(range(96), signature, tick_label=labels)
    plt.xticks(rotation=90)
    plt.show()

def get_data_batches(signatures, batch_size, n_samples=1000):
    input_batch = torch.empty(batch_size, 96)
    label_batch = torch.empty(batch_size, 72)

    for i in range(batch_size):
        label = random.randint(0, len(signatures)-1)
        signature = signatures[label]
        c = torch.distributions.categorical.Categorical(probs=signature)
        samples = c.sample(sample_shape=torch.Size([n_samples,])).type(torch.float32)
        sample = torch.histc(samples, bins=96, min=0, max=95)/float(n_samples)
        # plot_signature(signature=signature.numpy(), labels=None)
        # plot_signature(signature=sample.numpy(), labels=None)
        input_batch[i, :] = sample
        label_batch[i, :] = torch.nn.functional.one_hot(torch.tensor(label),  num_classes=72)

    return input_batch, label_batch