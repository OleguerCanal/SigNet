import torch
import torch.nn as nn

class SignatureNet(nn.Module):

    def __init__(self):
        super(SignatureNet, self).__init__()
        
        self.fc1 = nn.Linear(96, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 72)

        self.activation = torch.sigmoid


    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = nn.Softmax(dim=1)(x)
        return x
