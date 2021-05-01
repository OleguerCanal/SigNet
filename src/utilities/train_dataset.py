import torch
from torch.utils.data import Dataset, DataLoader


class TrainDataSet(Dataset):
    def __init__(self, train_input, train_label, train_baseline):
        self.train_input = train_input
        self.train_label = train_label
        self.train_baseline = train_baseline

    def __len__(self):
        return self.train_input.shape[0]

    def __getitem__(self, i):
        return self.train_input[i], self.train_label[i], self.train_baseline[i]


if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([5, 6, 7, 8])
    c = torch.tensor([9, 10, 11, 12])
    data = TrainingDataSet(a, b, c)
    train_dataloader = DataLoader(data, batch_size=2, shuffle=True)
    for a_batch, b_batch, c_batch in train_dataloader:
        print(a_batch)
        print(b_batch)
        print(c_batch)
        print("#######")