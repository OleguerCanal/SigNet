import torch
import torch.nn as nn

from baseline import SignatureFinder

class SignatureNet(nn.Module):

    def __init__(self, signatures, num_classes=72, num_hidden_layers=2, num_units=400):
        super(SignatureNet, self).__init__()
        self.num_classes = num_classes
        
        self.layers = [nn.Linear(num_classes + 96, num_units)]
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(num_units, num_units))
        self.layers.append(nn.Linear(num_units, num_classes))
        self.layers = nn.ModuleList(self.layers)

        self.activation = nn.LeakyReLU(0.1)

        self.model = SignatureFinder(signatures)
        self.signatures = torch.stack(signatures).t()


    def forward(self, x):
        w = self.model.get_weights_batch(x)
        guess = torch.einsum("ij,bj->bi", (self.signatures, w))
        error = x - guess
        x = torch.cat([w, error], dim=1)

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            #self.drop = nn.Dropout(p=0.5)
        x = self.layers[-1](x)
        # x = nn.Softmax(dim=1)(x)
        return x
