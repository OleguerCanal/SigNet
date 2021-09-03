import torch
import torch.nn as nn

class FineTuner(nn.Module):

    def __init__(self,
                 num_classes=72,
                 num_hidden_layers=2,
                 num_units=400,
                 cutoff=0.05):
        super(FineTuner, self).__init__()
        self._cutoff = cutoff

        # Num units of the mutations path
        num_units_branch_mut = int(num_units*0.1)

        # Num units of the other paths
        num_units_other_branches = num_units - num_units_branch_mut

        num_units_other_branches = int(num_units_other_branches/2)*2  # To have an even number of units
        num_units_branch = int(num_units_other_branches/2)

        self.layer1_1 = nn.Linear(num_classes, num_units_branch)    # Baseline guess path
        # 96 = total number of possible muts
        self.layer1_2 = nn.Linear(96, num_units_branch)             # Input path
        self.layer1_3 = nn.Linear(1, num_units_branch_mut)          # Number of mutations path

        self.layer2_1 = nn.Linear(num_units_branch, num_units_branch)
        self.layer2_2 = nn.Linear(num_units_branch, num_units_branch)
        self.layer2_3 = nn.Linear(num_units_branch_mut, num_units_branch_mut)

        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units, num_units)
                     for _ in range(num_hidden_layers)])

        self.output_layer = nn.Linear(num_units, num_classes)
        self.activation = nn.LeakyReLU(0.1)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.0001)

    def forward(self,
                mutation_dist,
                weights,
                num_mut):
        # Input head
        mutation_dist = self.activation(self.layer1_2(mutation_dist))
        mutation_dist = self.dropout(mutation_dist)
        mutation_dist = self.activation(self.layer2_2(mutation_dist))
        mutation_dist = self.dropout(mutation_dist)

        # Baseline head
        weights = self.activation(self.layer1_1(weights))
        weights = self.dropout(weights)
        weights = self.activation(self.layer2_1(weights))
        weights = self.dropout(weights)

        # Number of mutations head
        num_mut = self.activation(self.layer1_3(num_mut))
        num_mut = self.dropout(num_mut)
        num_mut = self.activation(self.layer2_3(num_mut))
        num_mut = self.dropout(num_mut)

        # Concatenate
        comb = torch.cat([mutation_dist, weights, num_mut], dim=1)

        # Apply shared layers
        for layer in self.hidden_layers:
            comb = self.activation(layer(comb))
            comb = self.dropout(comb)

        # Apply output layer
        comb = self.output_layer(comb)
        comb = self.softmax(comb)

        # If in eval mode, send small values to 0
        if not self.training:
            mask = (comb > self._cutoff).type(torch.int).float()
            comb = comb*mask
            comb = torch.div(comb,torch.sum(comb, axis=1).reshape((-1,1)))
        return comb
