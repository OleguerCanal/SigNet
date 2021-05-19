import torch
import torch.nn as nn

class SignedErrorFinder(nn.Module):
    def __init__(self,
                 num_classes=72,
                 num_hidden_layers=2,
                 num_units=400,
                 normalize_mut=2e4):
        super(SignedErrorFinder, self).__init__()
        self.normalize_mut = normalize_mut
        num_units = int(num_units/2)*2
        num_units_branch = int(num_units/2)

        # Weights path
        self.layer1_1 = nn.Linear(num_classes, num_units_branch)
        self.layer2_1 = nn.Linear(num_units_branch, num_units_branch)

        # Mutations path
        self.layer1_2 = nn.Linear(1, num_units_branch)

        # Combined layers
        self.hidden_layers = nn.ModuleList(
            modules=[nn.Linear(num_units, num_units)
                     for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(num_units, num_classes)

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
        num_mutations = self.activation(
            self.layer1_2(num_mutations/self.normalize_mut))

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
                 num_units_neg=400,
                 normalize_mut=20000):
        super(ErrorFinder, self).__init__()

        self.positive_path = SignedErrorFinder(num_classes=num_classes,
                                               num_hidden_layers=num_hidden_layers_pos,
                                               num_units=num_units_pos,
                                               normalize_mut=normalize_mut)

        self.negative_path = SignedErrorFinder(num_classes=num_classes,
                                               num_hidden_layers=num_hidden_layers_neg,
                                               num_units=num_units_neg,
                                               normalize_mut=normalize_mut)


    def forward(self, weights, num_mutations):
        pred_upper = self.positive_path(weights, num_mutations)
        pred_lower = self.negative_path(weights, num_mutations)
        return pred_upper, pred_lower
