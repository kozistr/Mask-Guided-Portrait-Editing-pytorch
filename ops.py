from functools import partial

import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type: str = 'instance', num_features: int = 64):
    if norm_type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'instance_layer':
        norm_layer = partial(ILN, num_features=num_features)
    else:
        raise NotImplementedError('normalization layer {} is not found'.format(norm_type))
    return norm_layer


class ILN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1.1e-5):
        super(ILN, self).__init__()

        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))

        self.rho.data.fill_(.0)
        self.gamma.data.fill_(1.)
        self.beta.data.fill_(0.)

    def forward(self, x):
        in_mean, in_var = \
            torch.mean(x, dim=(2, 3), keepdim=True), torch.var(x, dim=(2, 3), keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)

        ln_mean, ln_var = \
            torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)

        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1) + self.beta.expand(x.shape[0], -1, -1, -1)
        return out


class RhoClipper:
    def __init__(self, _min: float, _max: float):
        self.clip_min = _min
        self.clip_max = _max
        assert _min < _max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
