import torch
from torch import nn

class Generator(nn.Module):
    
    def __init__(self,
                 input_size=72,
                 num_hidden_layers=3,
                 latent_dim=50,
                 device="cuda") -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "Generator"
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.encoder_layers = nn.ModuleList(
            modules=[nn.Linear(int(input_size - (input_size-latent_dim)*i/num_hidden_layers), int(input_size - (input_size-latent_dim)*(i+1)/num_hidden_layers))
                     for i in range(num_hidden_layers)])
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.var_layer = nn.Linear(latent_dim, latent_dim)

        self.decoder_layers = nn.ModuleList(
            modules=[nn.Linear(int(input_size - (input_size-latent_dim)*(i+1)/num_hidden_layers), int(input_size - (input_size-latent_dim)*i/num_hidden_layers))
                     for i in reversed(range(num_hidden_layers))])

        self.activation = nn.LeakyReLU(0.1)

        self.Normal = torch.distributions.Normal(0, 1)
        if device == "cuda":
            self.Normal.loc = self.Normal.loc.cuda()
            self.Normal.scale = self.Normal.scale.cuda()


    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        z_mu = self.mean_layer(x)
        z_var = torch.exp(self.var_layer(x))
        return z_mu, z_var

    def decode(self, x):
        for layer in self.decoder_layers:
            x = self.activation(layer(x))
        return x

    def forward(self, x):
        z_mu, z_var = self.encode(x)
        z = z_mu + z_var*self.Normal.sample(z_mu.shape)
        x = self.decode(z)
        return x, z_mu, z_var

    def generate(self, batch_size:int):
        shape = tuple((batch_size, self.latent_dim))
        z = self.Normal.sample(shape)
        return self.decode(z)