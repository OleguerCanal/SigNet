import torch
from torch import nn

class Generator(nn.Module):
    
    def __init__(self,
                 input_size=72,
                 num_hidden_layers=2,
                 latent_dim=50,
                 num_cancers=37,  #TODO(Oleguer): Check this number
                 device="cuda") -> None:
        self.init_args = locals()
        self.init_args.pop("self")
        self.init_args.pop("__class__")
        self.init_args["model_type"] = "Generator"
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.num_cancers = num_cancers
        encoder_layers = []
        for i in range(num_hidden_layers):
            in_features = int(input_size - (input_size-latent_dim)*i/num_hidden_layers)
            out_features = int(input_size - (input_size-latent_dim)*(i+1)/num_hidden_layers)
            # print("in_features:", in_features, "out_features:", out_features)
            layer = nn.Linear(in_features, out_features)
            layernorm = torch.nn.LayerNorm(in_features)
            encoder_layers.append(layernorm)
            encoder_layers.append(layer)
        self.encoder_layers = nn.ModuleList(modules=encoder_layers)
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.var_layer = nn.Linear(latent_dim, latent_dim)

        decoder_layers = []
        for i in reversed(range(num_hidden_layers+1)):
            in_features = int(input_size - (input_size-latent_dim)*(i+1)/(num_hidden_layers + 1))
            out_features = int(input_size - (input_size-latent_dim)*i/(num_hidden_layers + 1))
            print("in_features:", in_features, "out_features:", out_features)
            # layernorm = torch.nn.LayerNorm(in_features)
            layer = nn.Linear(in_features, out_features)            
            # decoder_layers.append(layernorm)
            decoder_layers.append(layer)

        self.decoder_layers = nn.ModuleList(modules=decoder_layers)
        self.activation = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        

        self.Normal = torch.distributions.Normal(0, 1)
        if device == "cuda":
            print("Sending generator to cuda")
            self.Normal.loc = self.Normal.loc.cuda()
            self.Normal.scale = self.Normal.scale.cuda()
        self.device = device


    def __get_mask(self, cancer_ids):
        assert type(cancer_ids) == torch.Tensor
        lower = (cancer_ids*self.latent_dim/self.num_cancers).unsqueeze(dim=-1)
        upper = ((cancer_ids + 1)*self.latent_dim/self.num_cancers).unsqueeze(dim=-1)
        idx = torch.arange(self.latent_dim, device=self.device)
        mask = idx.ge(lower) & idx.le(upper)
        return mask.to(torch.int)


    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        z_mu = self.mean_layer(x)
        z_var = torch.exp(self.var_layer(x))
        # z_var = torch.ones_like(z_mu)
        # z_var = torch.ones_like(z_mu)*0.7
        # z_var = torch.ones_like(z_mu)*0.0001
        return z_mu, z_var


    def decode(self, x):
        for layer in self.decoder_layers[:-1]:
            x = self.activation(layer(x))
        x = self.relu(self.decoder_layers[-1](x))
        x = x/x.sum(dim=1).reshape(-1,1)
        return x

    def forward(self, x, cancer_ids=None, noise=True):
        z_mu, z_var = self.encode(x)
        if noise:
            # z = z_mu + z_var*self.Normal.sample(z_mu.shape)
            z = torch.randn(size = (z_mu.size(0), z_mu.size(1)), device=self.device)
            z = z_mu + z_var*z
        else:
            z = z_mu
        if cancer_ids is not None:
            mask = self.__get_mask(cancer_ids)
            z = z*mask
        x = self.decode(z)
        return x, z_mu, z_var

    def generate(self, batch_size:int, cancer_ids=None, std = 1.0):
        shape = tuple((batch_size, self.latent_dim))
        z = self.Normal.sample(shape)*std
        if cancer_ids is not None:
            mask = self.__get_mask(cancer_ids)
            z = z*mask
        labels = self.decode(z)
        return labels