import torch
from torch import nn

from .blocks.vision import DBlock


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1, 1)
        self.discriminator = nn.Sequential(DBlock(32, 64),
                                           DBlock(64, 128),
                                           DBlock(128, 256),
                                           DBlock(256, 256),
                                           DBlock(256, 256),
                                           DBlock(256, 256))

        self.classifier = nn.Sequential(nn.Conv2d(32*8+256, 32*2, 3, 1, 1, bias=False),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(32*2, 1, 4, 1, 0, bias=False))

    def forward(self, x, txt_emb):
        out = self.conv(x)
        out = self.discriminator(out)

        txt_emb = txt_emb.reshape(-1, 256, 1, 1).repeat(1, 1, 4, 4)
        feature = torch.cat((out, txt_emb), 1)
        pred = self.classifier(feature)

        return pred.flatten()

