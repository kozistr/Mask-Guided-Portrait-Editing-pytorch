import torch
import torch.nn as nn
from torchvision import models


class VGG19(nn.Module):
    def __init__(self, requires_grad: bool = False):
        super(VGG19, self).__init__()

        vgg_features = models.vgg19(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(1):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg_features[x])
        for x in range(20, 29):
            self.slice5.add_module(str(x), vgg_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        conv1 = self.slice1(x)
        conv2 = self.slice2(conv1)
        conv3 = self.slice3(conv2)
        conv4 = self.slice4(conv3)
        conv5 = self.slice5(conv4)
        out = [conv1, conv2, conv3, conv4, conv5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, weights: tuple = ()):
        super(VGGLoss, self).__init__()

        if weights:
            self.weights = weights
        else:
            self.weights = (1. / 4, 1. / 4, 1. / 4, 1. / 8, 1. / 8)

        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x, y, face_mask, mask_weights):
        assert face_mask.size()[1] == len(mask_weights)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        mask: list = [face_mask.detach()]
        for i in range(len(x_vgg)):
            mask.append(self.down_sample(mask[i]))
            mask[i] = mask[i].detach()

        loss = .0
        for i in range(len(x_vgg)):
            for mask_index in range(len(mask_weights)):
                loss += \
                    self.weights[i] * self.criterion(
                        x_vgg[i] * mask[i][:, mask_index:mask_index + 1, :, :],
                        (y_vgg[i] * mask[i][:, mask_index:mask_index + 1, :, :]).detach()) * mask_weights[mask_index]
        return loss


class MFMLoss(nn.Module):
    def __init__(self):
        super(MFMLoss, self).__init__()

        self.criterion = nn.MSELoss()

    def forward(self, x_input, y_input):
        loss = .0
        for i in range(len(x_input)):
            x = x_input[i][-2]
            y = y_input[i][-2]
            assert x.dim() == 4
            assert y.dim() == 4
            x_mean = torch.mean(x, 0)
            y_mean = torch.mean(y, 0)
            loss += self.criterion(x_mean, y_mean.detach())
        return loss


def gram_matrix(feature):
    assert feature.dim() == 4
    b, c, w, h = feature.size()[0], feature.size()[1], feature.size()[2], feature.size()[3]
    out_tensor = torch.Tensor(b, c, c).cuda()
    for batch_index in range(b):
        features = feature[batch_index].view(c, w * h)
        gram = torch.mm(features, features.t())
        out_tensor[batch_index] = gram.clone().div(c * w * h)
    return out_tensor


class GramMatrixLoss(nn.Module):
    def __init__(self):
        super(GramMatrixLoss, self).__init__()

        self.weights = [1., 1., 1.]
        self.vgg = VGG19().cuda()

        self.criterion = nn.MSELoss()
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x, y, label):
        face_mask = (label == 1).type(torch.cuda.FloatTensor)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        mask: list = [face_mask]
        for i in range(len(x_vgg)):
            mask.append(self.down_sample(mask[i]))
            mask[i] = mask[i].detach()

        loss = .0
        for i in range(len(x_vgg)):
            loss += \
                self.weights[i] * self.criterion(
                    gram_matrix(x_vgg[i] * mask[i]),
                    gram_matrix(y_vgg[i] * mask[i]).detach())
        return loss
