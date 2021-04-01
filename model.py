import torch
import torch.nn as nn

from baseline import SignatureFinder

class SignatureNet(nn.Module):

    def __init__(self, signatures, num_classes=72, num_hidden_layers=2, num_units=400):
        super(SignatureNet, self).__init__()
        self.num_classes = num_classes
        
        # self.layers = [nn.Linear(num_classes + 96, num_units)]
        # for _ in range(num_hidden_layers):
        #     self.layers.append(nn.Linear(num_units, num_units))
        # self.layers.append(nn.Linear(num_units, num_classes))
        # self.layers = nn.ModuleList(self.layers)

        self.layer1_1 = nn.Linear(num_classes, 128)
        self.layer1_2 = nn.Linear(96, 128)

        self.layer2_1 = nn.Linear(128, 256)
        self.layer2_2 = nn.Linear(128, 256)

        self.layer3 = nn.Linear(512,256)
        self.layer4 = nn.Linear(256,num_classes)
        self.activation = nn.LeakyReLU(0.1)

        #self.model = SignatureFinder(signatures)
        self.signatures = torch.stack(signatures).t()


    def forward(self, x, w):
        #w = self.model.get_weights_batch(x)
        # guess = torch.einsum("ij,bj->bi", (self.signatures, w))
        # error = x - guess
        # x = torch.cat([w, error], dim=1)
        # for layer in self.layers[:-1]:
        #     x = self.activation(layer(x))
        #     #self.drop = nn.Dropout(p=0.5)
        # x = self.layers[-1](x)
        # x = nn.Softmax(dim=1)(x)

        w = self.activation(self.layer1_1(w))
        x = self.activation(self.layer1_2(x))

        w = self.activation(self.layer2_1(w))
        x = self.activation(self.layer2_2(x))

        comb = torch.cat([w,x], dim=1)
        comb = self.activation(self.layer3(comb))
        comb = self.layer4(comb)
        comb = nn.Softmax(dim=1)(comb)
        return comb
