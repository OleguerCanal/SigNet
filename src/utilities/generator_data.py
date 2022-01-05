import torch
from torch.utils.data import Dataset

class GeneratorData(Dataset):
    def __init__(self, inputs=None):
        self.inputs = inputs

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, i):
        return self.inputs[i]
            