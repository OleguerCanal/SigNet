import torch
import torch.nn as nn

class FineTuner(nn.Module):

    def __init__(self,
                 num_classes=72,
                 num_hidden_layers=2,
                 num_units=400):
        super(FineTuner, self).__init__()

        num_units = int(num_units/2)*2  # To have an even number of units
        num_units_branch = int(num_units/2)

        self.layer1_1 = nn.Linear(num_classes, num_units_branch)
        # 96 = total number of possible muts
        self.layer1_2 = nn.Linear(96, num_units_branch)

        self.layer2_1 = nn.Linear(num_units_branch, num_units_branch)
        self.layer2_2 = nn.Linear(num_units_branch, num_units_branch)

        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units, num_units)
                     for _ in range(num_hidden_layers)])

        self.output_layer = nn.Linear(num_units, num_classes)
        self.activation = nn.LeakyReLU(0.1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self,
                mutation_dist,
                weights):
        # Input head
        mutation_dist = self.activation(self.layer1_2(mutation_dist))
        mutation_dist = self.activation(self.layer2_2(mutation_dist))

        # Baseline head
        weights = self.activation(self.layer1_1(weights))
        weights = self.activation(self.layer2_1(weights))

        # Concatenate
        comb = torch.cat([mutation_dist, weights], dim=1)

        # Apply shared layers
        for layer in self.hidden_layers:
            comb = self.activation(layer(comb))

        # Apply output layer
        comb = self.output_layer(comb)
        comb = self.softmax(comb)
        return comb
