import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid


class Classifier(nn.Module):

    def __init__(self,
                 num_hidden_layers=2,
                 num_units=400,
                 sigmoid_params = [5000,1000]):
        super(Classifier, self).__init__()
        self.sigmoid_params = sigmoid_params
        num_units_branch_mut = 10
        num_units_joined_path = num_units + num_units_branch_mut

        # Input path
        self.layer1_1 = nn.Linear(96, num_units)  # 96 = total number of possible muts
        # Number of mutations path
        self.layer1_2 = nn.Linear(1, num_units_branch_mut)

        self.layer2_1 = nn.Linear(num_units, num_units)
        self.layer2_2 = nn.Linear(num_units_branch_mut, num_units_branch_mut)

        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units_joined_path, num_units_joined_path)
                     for _ in range(num_hidden_layers)])

        self.output_layer = nn.Linear(num_units_joined_path, 1)
        self.activation = nn.LeakyReLU(0.1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,
                mutation_dist,
                num_mut):
        # Input head
        mutation_dist = self.activation(self.layer1_1(mutation_dist))
        mutation_dist = self.activation(self.layer2_1(mutation_dist))

        # Number of mutations head
        num_mut = nn.Sigmoid()((num_mut-self.sigmoid_params[0])/self.sigmoid_params[1])
        assert(not torch.isnan(num_mut).any())
        num_mut = self.activation(self.layer1_2(num_mut))
        num_mut = self.activation(self.layer2_2(num_mut))

        # Concatenate
        comb = torch.cat([mutation_dist, num_mut], dim=1)
        assert(not torch.isnan(comb).any())

        # Apply shared layers
        for layer in self.hidden_layers:
            comb = self.activation(layer(comb))

        # Apply output layer
        comb = self.output_layer(comb)
        comb = self.sigmoid(comb)

        return comb