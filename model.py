import torch
import torch.nn as nn

class SignatureNet(nn.Module):

    def __init__(self, num_classes=72, num_hidden_layers=2, num_units=400):
        super(SignatureNet, self).__init__()
        
        self.layers = [nn.Linear(96, num_units)]
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(num_units, num_units))
        self.layers.append(nn.Linear(num_units, num_classes))
        self.layers = nn.ModuleList(self.layers)

        self.activation = torch.nn.LeakyReLU(0.1)


    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            #self.drop = nn.Dropout(p=0.5)
        #x = nn.Softmax(dim=1)(x)
        return x
