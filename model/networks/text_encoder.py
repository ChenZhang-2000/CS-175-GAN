import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()