import torch
from torch import nn

from .blocks.vision import GBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 256*4*4)
        self.generator = nn.Sequential(GBlock(356, 256, 256),
                                       GBlock(356, 256, 256),
                                       GBlock(356, 256, 256),
                                       GBlock(356, 256, 128),
                                       GBlock(356, 128, 64),
                                       GBlock(356, 64, 32))

        self.decode = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(32, 3, 3, 1, 1),
                                    nn.Tanh())

    def forward(self, txt_emb):
        bs = txt_emb.shape[0]
        noise = torch.rand(bs, 100).cuda()
        out = self.fc(noise)
        out = out.reshape(bs, 256, 4, 4)

        y = torch.cat((noise, txt_emb), dim=1)

        out = self.generator((out, y))
        out = self.decode(out[0])

        return out