import torch
import torch.nn as nn


class SignedErrorFinder(nn.Module):
    def __init__(self,
                 num_classes=72,
                 num_hidden_layers=2,
                 num_units=400):
        super(SignedErrorFinder, self).__init__()
        # Num units of the weights path
        num_units_branch_1 = int(num_units*0.9)
        # Num units of the mutations path
        num_units_branch_2 = num_units - num_units_branch_1

        # Weights path
        self.layer1_1 = nn.Linear(num_classes, num_units_branch_1)
        nn.init.kaiming_uniform_(self.layer1_1.weight)
        self.layer2_1 = nn.Linear(num_units_branch_1, num_units_branch_1)
        nn.init.kaiming_uniform_(self.layer2_1.weight)

        # Mutations path
        self.layer1_2 = nn.Linear(1, num_units_branch_2)
        nn.init.kaiming_uniform_(self.layer1_2.weight)

        # Combined layers
        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units, num_units)
                     for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(num_units, num_classes)
        nn.init.kaiming_uniform_(self.output_layer.weight)
        for hidden_layer in self.hidden_layers:
            nn.init.kaiming_uniform_(hidden_layer.weight)

        


        # Activation function
        self.activation = nn.LeakyReLU(0.1)
        # self.activation = nn.ELU(0.1)

    def __clamp(self, x, slope=1e-2):
        return nn.LeakyReLU(slope)(1 - nn.LeakyReLU(slope)(1 - x))

    def forward(self, weights, num_mutations):
        # Baseline head
        weights = self.activation(self.layer1_1(weights))
        weights = self.activation(self.layer2_1(weights))

        # Mutations head
        num_mutations = nn.Sigmoid()((num_mutations-500)/150)  # Normalize
        num_mutations = self.activation(self.layer1_2(num_mutations))

        # Concatenate
        comb = torch.cat([weights, num_mutations], dim=1)

        # Apply shared layers
        for layer in self.hidden_layers:
            comb = self.activation(layer(comb))

        # Apply output layer
        # res = self.__clamp(self.output_layer(comb))
        res = self.output_layer(comb)
        return res


class ErrorFinder(nn.Module):

    def __init__(self,
                 num_classes=72,
                 num_hidden_layers_pos=2,
                 num_hidden_layers_neg=2,
                 num_units_pos=400,
                 num_units_neg=400):
        super(ErrorFinder, self).__init__()

        self.positive_path = SignedErrorFinder(num_classes=num_classes,
                                               num_hidden_layers=num_hidden_layers_pos,
                                               num_units=num_units_pos)

        self.negative_path = SignedErrorFinder(num_classes=num_classes,
                                               num_hidden_layers=num_hidden_layers_neg,
                                               num_units=num_units_neg)

    def forward(self, weights, num_mutations):
        pred_upper = self.positive_path(weights, num_mutations)
        pred_lower = self.negative_path(weights, num_mutations)
        return pred_upper, pred_lower
