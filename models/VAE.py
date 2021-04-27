import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, num_latent_dimensions, num_channels):
        super().__init__()

        self.num_latent_dimensions = num_latent_dimensions

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = num_channels

        ## Construct the encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=h_dim,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.latent_mean = nn.Linear(in_features=hidden_dims[-1],
                                     out_features=num_latent_dimensions)
        self.latent_sigma = nn.Linear(in_features=hidden_dims[-1],
                                      out_features=num_latent_dimensions)

        ## Construct the decoder
        modules = []
        self.decoder_input = nn.Linear(in_features=num_latent_dimensions,
                                       out_features=hidden_dims[-1])
        hidden_dims.reverse()

        kernel_sizes = [3, 3, 2, 2, 2]
        for i in range(len(hidden_dims)-1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_dims[i],
                                   out_channels=hidden_dims[i+1],
                                   kernel_size=kernel_sizes[i],
                                   stride=2),
                                   #padding=1,
                                   #output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU()
            ))

        self.decoder = nn.Sequential(*modules)

        ## Add a final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1],
                               out_channels=hidden_dims[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[-1],
                      out_channels=1,
                      kernel_size=3,
                      padding=1),
            nn.Tanh())

    def encode(self, input):
        result = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)
        result = result.view(result.size(0), -1)
        mean = self.latent_mean(result)
        log_var = self.latent_sigma(result)
        return [mean, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = result.view(result.size(0), 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mean, log_var):
        std = torch.exp(log_var)
        eps = torch.randn_like(std)
        return mean + eps*std

    def forward(self, input):
        [mean, log_var] = self.encode(input)
        z = self.reparameterize(mean, log_var)
        result = self.decode(z)
        return result, mean, log_var

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.num_latent_dimensions)
        samples = self.decode(z)
        return samples
