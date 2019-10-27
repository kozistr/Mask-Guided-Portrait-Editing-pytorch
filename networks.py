import torch
import numpy as np
import torch.nn as nn

from .ops import get_norm_layer
from .ops import weights_init


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, use_bias: bool = False):
        super(ResNetBlock, self).__init__()

        norm_layer = get_norm_layer('instance')

        conv_block: list = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(dim)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x, scale: float = 1.):
        out = scale * x + self.conv_block(x)
        return out


class ResNetAdaILNBlock(nn.Module):
    def __init__(self, dim: int, use_bias: bool = False):
        super(ResNetAdaILNBlock, self).__init__()

        norm_layer = get_norm_layer('adaptive_instance_layer', num_features=dim)

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = norm_layer
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = norm_layer

    def forward(self, x, gamma, beta, scale: float = 1.):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + scale * x


class EncoderBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int,
                 kernel_size: int = 7, pad: int = 3, stride: int = 2, norm_type: str = 'instance'):
        super(EncoderBlock, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)

        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=stride)
        self.norm = norm_layer(ch_out)
        self.act = nn.ReLU(True)

    def forward(self, x, inter: bool = False):
        x = self.pad(x)
        x = self.conv(x)
        x_inter = x if inter else None
        x = self.norm(x)
        x = self.act(x)
        return x, x_inter


class DecoderBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int,
                 kernel_size: int = 4, pad: int = 1, stride: int = 2, norm_type: str = 'instance'):
        super(DecoderBlock, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)

        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=kernel_size, stride=stride)
        self.norm = norm_layer(ch_out)
        self.act = nn.ReLU(True)

    def forward(self, x, use_act: bool = True):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        if use_act:
            x = self.act(x)
        return x


# Component Encoders used at Local Embedding Sub-Network

class ComponentEncoder(nn.Module):
    """ Encoding components. Each Component has a shape of
    Eye               : (3,  32,  48)
    Mouth             : (3,  80, 144)
    Skin (Face, Hair) : (3, 256, 256)
    """

    def __init__(self, input_shape: tuple = (3, 256, 256), norm_type: str = 'instance',
                 n_feat: int = 64, ch_emb: int = 512, inter_fc_units: int = 1024):
        super(ComponentEncoder, self).__init__()

        c, w, h = input_shape
        curr_feat: int = n_feat
        n_repeats = int(np.log2(ch_emb) - np.log2(curr_feat))

        assert w in [32, 80, 256]

        layers: list = [EncoderBlock(c, curr_feat, kernel_size=4, pad=1, stride=2)]
        for _ in range(n_repeats):
            layers.append(EncoderBlock(curr_feat, curr_feat * 2, kernel_size=4, pad=1, stride=2, norm_type=norm_type))
            curr_feat *= 2

        if w == 256:  # in case of skin (face, hair)
            for _ in range(3):
                layers.append(EncoderBlock(curr_feat, curr_feat, kernel_size=4, pad=1, stride=2, norm_type=norm_type))

        w_enc: int = w // (2 ** len(layers))
        h_enc: int = h // (2 ** len(layers))

        self.model = nn.Sequential(*layers)

        self.fc_mu = nn.Sequential(
            nn.Linear(ch_emb * w_enc * h_enc, inter_fc_units),
            nn.ReLU(True),
            nn.Linear(inter_fc_units, ch_emb)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(ch_emb * w_enc * h_enc, inter_fc_units),
            nn.ReLU(True),
            nn.Linear(inter_fc_units, ch_emb)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size()[0], -1)
        mu, var = self.fc_mu(x), self.fc_var(x)
        return mu, var


# Component Encoders used at Local Embedding Sub-Network

class ComponentDecoder(nn.Module):
    """ Decoding components. Each Component has a shape of
    Eye               : (3,  32,  48)
    Mouth             : (3,  80, 144)
    Skin (Face, Hair) : (3, 256, 256)
    """

    def __init__(self, input_shape: tuple = (3, 256, 256), norm_type: str = 'instance',
                 n_feat: int = 64, ch_emb: int = 512):
        super(ComponentDecoder, self).__init__()

        c, w, h = input_shape
        self.ch_emb = ch_emb
        curr_feat: int = self.ch_emb
        n_repeats = int(np.log2(self.ch_emb) - np.log2(n_feat))

        assert w in [32, 80, 256]

        layers: list = []
        if w == 256:  # in case of skin (face, hair)
            for _ in range(4):
                layers.append(DecoderBlock(curr_feat, curr_feat, kernel_size=4, pad=1, stride=2, norm_type=norm_type))

        for _ in range(n_repeats):
            layers.append(DecoderBlock(curr_feat, curr_feat // 2, kernel_size=4, pad=1, stride=2, norm_type=norm_type))
            curr_feat //= 2

        if w != 256:  # in case of skin (mouth, eye)
            layers.append(DecoderBlock(curr_feat, curr_feat, kernel_size=4, pad=1, stride=2, norm_type=norm_type))

        # len(layers) -> number of decoder blocks
        self.w_enc: int = w // (2 ** len(layers))
        self.h_enc: int = h // (2 ** len(layers))

        layers += [
            nn.ReflectionPad2d(2),
            nn.Conv2d(curr_feat, c, kernel_size=5, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(ch_emb, ch_emb * self.w_enc * self.h_enc),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size()[0], self.ch_emb, self.w_enc, self.h_enc)
        x = self.model(x)
        return x


# Component Projector used at Mask-Guided Generative Sub-Network

class ComponentProjector(nn.Module):
    """ Projecting component's embedding. Each projected component's embedding has a shape of
    Eye               : (256,  8, 12)
    Mouth             : (256, 20, 36)
    Skin (Face, Hair) : (256, 64, 64)
    """

    def __init__(self, input_shape: tuple = (3, 256, 256), norm_type: str = 'instance',
                 ch_emb: int = 512, ch_proj: int = 256):
        super(ComponentProjector, self).__init__()

        c, w, h = input_shape
        self.ch_proj = ch_proj
        curr_feat: int = ch_emb
        n_repeats = int(np.log2(ch_emb) - np.log2(self.ch_proj))

        assert w in [32, 80, 256]

        layers: list = []
        for _ in range(n_repeats):
            layers.append(DecoderBlock(curr_feat, curr_feat // 2, kernel_size=4, pad=1, stride=2, norm_type=norm_type))
            curr_feat //= 2

        if w == 256:  # in case of skin (face, hair)
            for _ in range(3):
                layers.append(DecoderBlock(curr_feat, curr_feat, kernel_size=4, pad=1, stride=2, norm_type=norm_type))

        layers.append(DecoderBlock(curr_feat, curr_feat, kernel_size=4, pad=1, stride=2, norm_type=norm_type))

        self.model = nn.Sequential(*layers)

        self.w_enc: int = w // (2 ** (len(layers) + 2))
        self.h_enc: int = h // (2 ** (len(layers) + 2))

        self.fc = nn.Sequential(
            nn.Linear(ch_emb, ch_emb * self.w_enc * self.h_enc, bias=False),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size()[0], self.ch_proj, self.w_enc, self.h_enc)
        x = self.model(x)
        return x


def build_component_encoder(input_shape: tuple = (3, 256, 256), norm_type: str = 'instance'):
    comp_encoder = ComponentEncoder(input_shape=input_shape, norm_type=norm_type)
    if torch.cuda.is_available():
        comp_encoder = comp_encoder.cuda()
    comp_encoder.apply(weights_init)
    return comp_encoder


def build_component_decoder(input_shape: tuple = (3, 256, 256), norm_type: str = 'instance'):
    comp_decoder = ComponentDecoder(input_shape=input_shape, norm_type=norm_type)
    if torch.cuda.is_available():
        comp_decoder = comp_decoder.cuda()
    comp_decoder.apply(weights_init)
    return comp_decoder


def build_component_projector(input_shape: tuple = (3, 256, 256), norm_type: str = 'instance'):
    comp_projector = ComponentProjector(input_shape=input_shape, norm_type=norm_type)
    if torch.cuda.is_available():
        comp_projector = comp_projector.cuda()
    comp_projector.apply(weights_init)
    return comp_projector


# Discriminators

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_feats: int = 64, n_layers: int = 3, norm_type: str = 'instance',
                 use_sigmoid: bool = False, num_disc: int = 3):
        super(MultiScaleDiscriminator, self).__init__()

    def forward(self, x):
        return x
