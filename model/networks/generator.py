import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()