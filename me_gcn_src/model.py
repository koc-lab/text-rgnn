import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(
        self, in_features, out_features, drop_out=0, activation=None, bias=True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters(in_features, out_features)
        self.dropout = torch.nn.Dropout(drop_out)
        self.activation = activation

    def reset_parameters(self, in_features, out_features):
        stdv = np.sqrt(6.0 / (in_features + out_features))
        # stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     torch.nn.init.zeros_(self.bias)
        # self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, feature_less=False):
        if feature_less:
            support = self.weight
        else:
            x = self.dropout(x)
            support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MULTIGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()

        # different weights
        self.intras1 = nn.ModuleList(
            [
                GraphConvolution(nfeat, nhid, dropout, activation=nn.ReLU())
                for i in range(25)
            ]
        )
        self.intras2 = nn.ModuleList(
            [
                GraphConvolution(nhid * 25, nclass, dropout, activation=nn.ReLU())
                for i in range(25)
            ]
        )

    def forward(self, x, adj, feature_less=False):
        x = torch.stack([self.intras1[i](x, adj[i], feature_less) for i in range(25)])
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size()[0], -1)
        x = torch.stack([self.intras2[i](x, adj[i]) for i in range(25)])
        return torch.max(x, 0)[0]
