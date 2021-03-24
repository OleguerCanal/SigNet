import torch
import torch.nn as nn

class SignatureNet(nn.Module):

    def __init__(self, num_hidden_layers=2, num_units=400):
        super(SignatureNet, self).__init__()
        
        self.layers = [nn.Linear(96, num_units)]
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(num_units, num_units))
        self.layers.append(nn.Linear(num_units, 72))

        self.activation = torch.ReLU()


    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = nn.Softmax(dim=1)(x)
        return x
