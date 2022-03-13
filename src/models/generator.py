import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self,
                 input_size=72,
                 num_hidden_layers=2,
                 latent_dim=50,
                 device="cuda") -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "Generator"
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        encoder_layers = []
        for i in range(num_hidden_layers):
            in_features = int(input_size - (input_size-latent_dim)*i/num_hidden_layers)
            out_features = int(input_size - (input_size-latent_dim)*(i+1)/num_hidden_layers)
            print("in_features:", in_features, "out_features:", out_features)
            layer = nn.Linear(in_features, out_features)            
            encoder_layers.append(layer)
        self.encoder_layers = nn.ModuleList(modules=encoder_layers)
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.var_layer = nn.Linear(latent_dim, latent_dim)
        self.activation = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        z_mu = self.mean_layer(x)
        z_log_var = self.var_layer(x)
        z_std = torch.exp(0.5*z_log_var)
        return z_mu, z_std

class Decoder(nn.Module):
    """This module is used both in the VAE and GAN versions of the generator
    """
    def __init__(self,
                 input_size=72,
                 latent_dim=50,
                 num_hidden_layers=2,
                 device="cuda") -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "Decoder"
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.device = device
        decoder_layers = []
        for i in reversed(range(num_hidden_layers+1)):
            in_features = int(input_size - (input_size-latent_dim)*(i+1)/(num_hidden_layers + 1))
            out_features = int(input_size - (input_size-latent_dim)*i/(num_hidden_layers + 1))
            print("in_features:", in_features, "out_features:", out_features)
            layer = nn.Linear(in_features, out_features)            
            decoder_layers.append(layer)

        self.decoder_layers = nn.ModuleList(modules=decoder_layers)
        self.activation = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

        self.Normal = torch.distributions.Normal(0, 1)
        if device == "cuda":
            print("sending to cuda")
            self.Normal.loc = self.Normal.loc.cuda()
            self.Normal.scale = self.Normal.scale.cuda()

    def forward(self, x):
        for layer in self.decoder_layers[:-1]:
            x = self.activation(layer(x))
        x = self.relu(self.decoder_layers[-1](x))
        x = x/x.sum(dim=1).reshape(-1,1)
        return x

    def sample(self, batch_size=1, mean=0, std=1):
        z = mean + std*torch.randn(batch_size, self.latent_dim, requires_grad=True).to(self.device)
        return self.forward(z)

class Discriminator(nn.Module):
    """This module is used in the GAN version of the generator
    """
    def __init__(self,
                 input_size=72,
                 num_layers=3):
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "Discriminator"
        super(Discriminator, self).__init__()
        layers = [nn.Linear(input_size, input_size) for _ in range(num_layers)]
        self.layers = nn.ModuleList(modules=layers)
        self.output_layer = nn.Linear(input_size, 1)
        self.out_act = nn.Tanh()
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.out_act(self.output_layer(x))

class Generator(nn.Module):
    
    def __init__(self,
                 input_size=72,
                 num_hidden_layers=2,
                 latent_dim=50,
                 device="cuda") -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "Generator"
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.device = device
        self.encoder = Encoder(input_size=input_size,
                               num_hidden_layers=num_hidden_layers,
                               latent_dim=latent_dim,
                               device=device)

        self.decoder = Decoder(input_size=input_size,
                               num_hidden_layers=num_hidden_layers,
                               latent_dim=latent_dim,
                               device=device)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, noise=True):
        z_mu, z_std = self.encode(x)
        if noise:
            # z = z_mu + z_std*self.Normal.sample(z_mu.shape)
            z = torch.randn(size = (z_mu.size(0), z_mu.size(1)), device=self.device)
            z = z_mu + z_std*z
        else:
            z = z_mu
        x = self.decode(z)
        return x, z_mu, z_std

    def generate(self, batch_size:int, std = 1.0):
        shape = tuple((batch_size, self.latent_dim))
        z = self.Normal.sample(shape)*std
        labels = self.decode(z)
        return labels

    def filter(self, syntethic_data, real_labels, quantile=0.75, print_dist_stats=False):
        """Remove outliers of synthetic_data by omitting
           (1 - quantile)% most-different points from the original dataset
        """
        def min_dist(point):
            return ((real_labels - point).pow(2)).mean(dim=1).min()
        with torch.no_grad():
            distances = torch.tensor([min_dist(p) for p in syntethic_data])

        if print_dist_stats:
            print("Min dist:", distances.min())
            print("Mean dist:", distances.mean())
            print("Max dist:", distances.max())

        quantiles = torch.quantile(distances, torch.tensor([quantile]), keepdim=True)
        accepted = distances < quantiles.flatten()[-1].item()
        return syntethic_data[accepted, ...]