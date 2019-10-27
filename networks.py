import torch.nn as nn

from .ops import get_norm_layer


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
