import numpy as np
import torch.nn as nn


"""
The code is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.img_shape = (opt.channels, opt.img_width, opt.img_height)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 1024, normalize=False),
            # *block(512, 1024, normalize=False),
            *block(1024, 2048, normalize=False),
            *block(2048, 4096, normalize=False),
            *block(4096, 8192),
            # *block(8192, 16384),
            # *block(4096, 8192),
            nn.Linear(8192, int(np.prod(self.img_shape))),
            nn.Tanh()
            )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        img_shape = (opt.channels, opt.img_width, opt.img_height)

        self.features = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 2048),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(8192, 4096),
            # nn.Linear(4096, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.last_layer = nn.Sequential(
            nn.Linear(256, 1)
            )

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.last_layer(features)
        return validity

    def forward_features(self, img):
        img_flat = img.view(img.shape[0], -1)
        features = self.features(img_flat)
        return features


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        img_shape = (opt.channels, opt.img_width, opt.img_height)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 2048),  
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(8192, 4096),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, opt.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
