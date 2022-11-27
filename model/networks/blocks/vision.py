from collections import OrderedDict

import torch
from torch import nn


class Affine(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([('linear1', nn.Linear(in_dim, out_dim)),
                                                   ('relu1', nn.ReLU(inplace=True)),
                                                   ('linear2', nn.Linear(out_dim, out_dim))]))
        self.fc_beta = nn.Sequential(OrderedDict([('linear1', nn.Linear(in_dim, out_dim)),
                                                  ('relu1', nn.ReLU(inplace=True)),
                                                  ('linear2', nn.Linear(out_dim, out_dim)),]))

        self.activate = nn.LeakyReLU(0.2, inplace=True)

        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, t):
        x, y = t
        w = self.fc_gamma(y)
        w = w.reshape(1, -1) if w.dim() == 1 else w

        b = self.fc_beta(y)
        b = b.reshape(1, -1) if b.dim() == 1 else b

        w = w.unsqueeze(-1).unsqueeze(-1).expand(x.shape)
        b = b.unsqueeze(-1).unsqueeze(-1).expand(x.shape)

        return self.activate(w * x + b), y


class GBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GBlock, self).__init__()
        self.learnable_sc = hidden_dim != out_dim
        self.conv1 = nn.Conv2d(hidden_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.affine1 = nn.Sequential(Affine(in_dim, hidden_dim),
                                     Affine(in_dim, hidden_dim))
        self.affine2 = nn.Sequential(Affine(in_dim, out_dim),
                                     Affine(in_dim, out_dim))

        if self.learnable_sc:
            self.shortcut = nn.Conv2d(hidden_dim, out_dim, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, t):
        x, y = t
        x = nn.functional.interpolate(x, scale_factor=2)

        out = self.affine1((x, y))
        out = self.conv1(out[0])
        out = self.affine2((out, y))
        out = self.conv2(out[0])
        out = self.shortcut(x) + out

        return out, y


class DBlock(nn.Module):
    def __init__(self, cin, cout):
        super(DBlock, self).__init__()
        self.res = nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1, bias=False),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.shortcut = nn.Conv2d(cin, cout, 1, 1, 0) if (cin != cout) else nn.Identity()
        self.k = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        shortcut = nn.functional.avg_pool2d(shortcut, 2)
        return self.k * self.res(x) + shortcut
