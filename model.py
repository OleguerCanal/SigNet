import torch
import torch.nn as nn

from baseline import SignatureFinder

class SignatureNet(nn.Module):

    def __init__(self, signatures=None, num_classes=72, num_hidden_layers=2, num_units=400):
        super(SignatureNet, self).__init__()
        self.num_classes = num_classes

        num_units = int(num_units/2)*2
        num_units_branch = int(num_units/2)
        self.layer1_1 = nn.Linear(num_classes, num_units_branch)
        self.layer1_2 = nn.Linear(1, num_units_branch)

        self.layer2_1 = nn.Linear(num_units_branch, num_units_branch)
        self.layer2_2 = nn.Linear(num_units_branch, num_units_branch)

        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units, num_units)
            for _ in range(num_hidden_layers)])

        self.output_layer = nn.Linear(num_units, num_classes)
        self.activation = nn.LeakyReLU(0.1)

        self.softmax = nn.Softmax(dim=1)
        #self.model = SignatureFinder(signatures)
        if signatures is not None:
            self.signatures = torch.stack(signatures).t()


    def forward(self, w, m):
        # Baseline head
        w = self.activation(self.layer1_1(w))
        w = self.activation(self.layer2_1(w))

        # Mutations head
        m = self.activation(self.layer1_2(m/20000))
        m = self.activation(self.layer2_2(m))

        # Concatenate
        comb = torch.cat([w, m], dim=1)

        # Apply shared layers
        for layer in self.hidden_layers:
            comb = self.activation(layer(comb))

        # Apply output layer
        comb = self.output_layer(comb)
        comb = self.softmax(comb)
        return comb
