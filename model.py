import torch
import torch.nn as nn

from baseline import SignatureFinder

class SignatureNet(nn.Module):

    def __init__(self, signatures=None, num_classes=72, num_hidden_layers_pos=2, num_hidden_layers_neg=2, num_units_pos=400, num_units_neg=400, normalize_mut = 20000):
        super(SignatureNet, self).__init__()
        self.num_classes = num_classes
        self.normalize_mut = normalize_mut

        # Positive error:
        num_units_pos = int(num_units_pos/2)*2
        num_units_branch_pos = int(num_units_pos/2)
        self.layer1_1_pos = nn.Linear(num_classes, num_units_branch_pos)
        self.layer1_2_pos = nn.Linear(1, num_units_branch_pos)

        self.layer2_1_pos = nn.Linear(num_units_branch_pos, num_units_branch_pos)
        self.layer2_2_pos = nn.Linear(num_units_branch_pos, num_units_branch_pos)

        self.hidden_layers_pos = nn.ModuleList(
            modules=[nn.Linear(num_units_pos, num_units_pos)
            for _ in range(num_hidden_layers_pos)])

        self.output_layer_pos = nn.Linear(num_units_pos, num_classes)

        # Negative error:
        num_units_neg = int(num_units_neg/2)*2
        num_units_branch_neg = int(num_units_neg/2)
        self.layer1_1_neg = nn.Linear(num_classes, num_units_branch_neg)
        self.layer1_2_neg = nn.Linear(1, num_units_branch_neg)

        self.layer2_1_neg = nn.Linear(num_units_branch_neg, num_units_branch_neg)
        self.layer2_2_neg = nn.Linear(num_units_branch_neg, num_units_branch_neg)

        self.hidden_layers_neg = nn.ModuleList(
            modules=[nn.Linear(num_units_neg, num_units_neg)
            for _ in range(num_hidden_layers_neg)])

        self.output_layer_neg = nn.Linear(num_units_neg, num_classes)

        self.activation = nn.LeakyReLU(0.1)

        self.softmax = nn.Softmax(dim=1)
        #self.model = SignatureFinder(signatures)
        if signatures is not None:
            self.signatures = torch.stack(signatures).t()


    def forward(self, w, m):
        # Positive error
        # Baseline head
        w_pos = self.activation(self.layer1_1_pos(w))
        w_pos = self.activation(self.layer2_1_pos(w_pos))

        # Mutations head
        m_pos = self.activation(self.layer1_2_pos(m/self.normalize_mut))
        m_pos = self.activation(self.layer2_2_pos(m_pos))

        # Concatenate
        comb_pos = torch.cat([w_pos, m_pos], dim=1)

        # Apply shared layers
        for layer in self.hidden_layers_pos:
            comb_pos = self.activation(layer(comb_pos))

        # Apply output layer
        res_pos = self.output_layer_pos(comb_pos)

        # Negative error
        # Baseline head
        w_neg = self.activation(self.layer1_1_neg(w))
        w_neg = self.activation(self.layer2_1_neg(w_neg))

        # Mutations head
        m_neg = self.activation(self.layer1_2_neg(m/self.normalize_mut))
        m_neg = self.activation(self.layer2_2_neg(m_neg))

        # Concatenate
        comb_neg = torch.cat([w_neg, m_neg], dim=1)

        # Apply shared layers
        for layer in self.hidden_layers_neg:
            comb_neg = self.activation(layer(comb_neg))

        # Apply output layer
        res_neg = self.output_layer_neg(comb_neg)

        #comb = self.softmax(comb)
        return res_pos, res_neg
