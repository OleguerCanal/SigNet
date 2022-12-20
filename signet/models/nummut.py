import torch.nn as nn
import torch.nn.functional as F


class NumMutNet(nn.Module):
    def __init__(self,
                 hidden_dim,
                 n_layers = 3,
                 input_dim = 72,
                 output_dim = 9):
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "NumMutNet"

        super(NumMutNet, self).__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.25)
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dropout(self.activation(self.input(x)))
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        x = self.output(x)
        return x

