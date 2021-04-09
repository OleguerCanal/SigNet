import torch
import torch.nn as nn

from baseline import SignatureFinder

class SignatureNet(nn.Module):

    def __init__(self, signatures, num_classes=72, num_hidden_layers=2, num_units=400):
        super(SignatureNet, self).__init__()
        self.num_classes = num_classes

        self.layer1_1 = nn.Linear(num_classes, 128)
        self.layer1_2 = nn.Linear(1, 16)

        self.layer2_1 = nn.Linear(128, 256)
        self.layer2_2 = nn.Linear(16, 256)

        self.layer3_1 = nn.Linear(512,512)
        self.layer3_2 = nn.Linear(512,256)

        self.layer4 = nn.Linear(256,num_classes)
        self.activation = nn.LeakyReLU(0.1)

        #self.model = SignatureFinder(signatures)
        self.signatures = torch.stack(signatures).t()


    def forward(self, w, m):

        w = self.activation(self.layer1_1(w))
        m = self.activation(self.layer1_2(m/2000))

        w = self.activation(self.layer2_1(w))
        m = self.activation(self.layer2_2(m))

        comb = torch.cat([w, m], dim=1)
        comb = self.activation(self.layer3_1(comb))
        comb = self.activation(self.layer3_2(comb))
        comb = self.layer4(comb)

        return comb
