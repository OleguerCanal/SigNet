import torch
from torch.utils.data import Dataset, DataLoader


class DataPartitions(Dataset):
    def __init__(self, inputs=None, labels=None, prev_guess=None, num_mut=None):
        """Encapsulates input, label and previous step guess of a dataset.
        
        Args:
            inputs (Torch.Tensor, optional): Input to current ANN. Defaults to None.
            labels (torch.Tensor, optional): Labels of current inputs. Defaults to None.
            prev_guess (Guess of the previous model, optional): If training finetuner, 
                this is the baseline output, if training the errorfinder this is the
                finetuner output. Defaults to None.
        """
        
        self.inputs = inputs
        self.labels = labels
        self.prev_guess = prev_guess
        self.num_mut = None
        if num_mut is not None:
            self.num_mut = num_mut
        if labels is not None:
            self.num_mut = labels[:, -1].reshape(-1,1)
            self.labels = labels[:, :-1]


    def to(self, device):
        device = torch.device(device)
        self.inputs = self.inputs.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.prev_guess is not None:
            self.prev_guess = self.prev_guess.to(device)
        if self.num_mut is not None:
            self.num_mut = self.num_mut.to(device)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, i):
        num_mut_i = None if self.num_mut is None else self.num_mut[i]
        if self.labels == None:
            return self.inputs[i], num_mut_i
        else:
            labels_i = None if self.labels is None else self.labels[i]
            prev_guess_i = None if self.prev_guess is None else self.prev_guess[i]
            return self.inputs[i], labels_i, prev_guess_i, num_mut_i