import torch.nn as nn
from dgl.nn.pytorch import GraphConv, SumPooling


class GCN(nn.Module):
    def __init__(self,
                 flen,
                 dropout_rate,
                 intermediate_rep=128,
                 n_hidden=256,
                 n_layers=6):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(flen, n_hidden, activation=nn.ReLU()))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=nn.ReLU()))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_hidden))
        self.linear_model = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),

            nn.Linear(n_hidden, intermediate_rep)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pooling = SumPooling()

    def forward(self, g, h):
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        h = self.pooling(g, h)
        h = self.linear_model(h)
        return h
