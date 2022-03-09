import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self,
                 input_size=72,
                 output_size=72,
                 num_hidden_layers=2,
                 device="cuda"):
        assert num_hidden_layers > 0
        super(Generator, self).__init__()
        self.input_size = input_size
        self.input_layer = nn.Linear(input_size, output_size)
        layers = [nn.Linear(output_size, output_size) for _ in range(num_hidden_layers)]
        self.layers = nn.ModuleList(modules=layers)
        self.relu = nn.ReLU()
        self.activation = nn.LeakyReLU(0.1)
        self.device = device

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.relu(self.layers[-1](x))
        x = x/x.sum(dim=1).reshape(-1,1)
        return x

    def sample(self, batch_size=1, mean=0, std=1):
        z = mean + std*torch.randn(batch_size, self.input_size, requires_grad=True).to(self.device)
        return self.forward(z)


class Discriminator(nn.Module):
    def __init__(self,
                 input_size=72,
                 num_layers=3):
        super(Discriminator, self).__init__()
        layers = [nn.Linear(input_size, input_size) for _ in range(num_layers)]
        self.layers = nn.ModuleList(modules=layers)
        self.output_layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.sigmoid(self.output_layer(x))


class GAN(nn.Module):
    
    def __init__(self,
                 num_classes=72,
                 generator_input_size=72,
                 generator_num_hidden_layers=2,
                 discriminator_num_hidden_layers=2,
                 device="cuda") -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("device")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "GAN"
        super(GAN, self).__init__()
        self.device = device

        self.generator = Generator(input_size=generator_input_size,
                                   output_size=num_classes,
                                   num_hidden_layers=generator_num_hidden_layers,
                                   device=device)
        
        self.discriminator = Discriminator(input_size=num_classes,
                                           num_layers=discriminator_num_hidden_layers)

    def _shuffle(self, x, y):
        indexes = torch.randperm(x.size(0))
        return x[indexes, ...], y[indexes, ...]

    def forward(self, real_inputs, origin):
        batch_size = real_inputs.size(0)

        if origin == "real":
            ones = torch.ones((batch_size, 1), dtype=torch.float).to(self.device)
            return self.discriminator(real_inputs), ones
        elif origin == "mixed":
            synt_inputs = self.generator.sample(batch_size=batch_size)
            merged_inputs = torch.cat([real_inputs, synt_inputs], dim=0)
            
            # Create real/fake labels (real=1, fake=0)
            ones = torch.ones((batch_size, 1), dtype=torch.float).to(self.device)
            zeros = torch.zeros((batch_size, 1), dtype=torch.float).to(self.device)
            labels = torch.cat([ones, zeros], dim=0)

            # Shuffle stuff (NOTE(oleguer): I dont think this step is necessary actually)
            merged_inputs, labels = self._shuffle(merged_inputs, labels)
            return self.discriminator(merged_inputs), labels
        elif origin == "fake":
            # Append synt inputs
            synt_inputs = self.generator.sample(batch_size=batch_size)
            zeros = torch.zeros((batch_size, 1), dtype=torch.float).to(self.device)

            return self.discriminator(synt_inputs), zeros
        raise  ValueError("Origin should be: real, mixed or fake :)")