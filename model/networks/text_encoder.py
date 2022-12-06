import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, in_dim=300, hiddem_dim=128, num_layers=1):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(0.2)
        self.in_dim = in_dim
        self.hiddem_dim = hiddem_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.in_dim, self.hiddem_dim, self.num_layers,
                            batch_first=True, dropout=0.2, bidirectional=True)
        self.embedding = nn.Embedding(27297, self.in_dim)

    def init_hidden(self, bs):
        return (Variable(torch.zeros(self.num_layers * 2,
                                     bs, self.hiddem_dim)).cuda(),
                Variable(torch.zeros(self.num_layers * 2,
                                     bs, self.hiddem_dim)).cuda())

    def forward(self, x, length, mask=None):
        x = self.drop(self.embedding(x))
        bs = x.shape[0]
        # print(x.shape)
        initial_hidden = self.init_hidden(bs)
        out, hid = self.lstm(pack_padded_sequence(x, length, batch_first=True), initial_hidden)
        # print(type(out))

        out = pad_packed_sequence(out, batch_first=True)[0].transpose(1, 2)

        sent_emb = hid[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, 256)

        return sent_emb # .flatten()