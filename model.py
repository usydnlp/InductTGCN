import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):

    def __init__(self, in_features, out_features,  drop_out = 0, activation=None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(in_features, out_features)
        self.dropout = torch.nn.Dropout(drop_out)
        self.activation =  activation

    def reset_parameters(self,in_features, out_features):
        stdv = np.sqrt(6.0/(in_features+out_features))
        # stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     torch.nn.init.zeros_(self.bias)
            # self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj, feature_less = False):
        if feature_less:           
            support1 = torch.mm(input, self.weight)
            support2 = self.weight
            support = torch.vstack((support1,support2))
            support = self.dropout(support)
        else:    
            input = self.dropout(input)
            support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, dropout, activation = nn.ReLU())
        self.gc2 = GraphConvolution(nhid, nclass, dropout)

    def forward(self, x, adj):
        x1 = self.gc1(x, adj, feature_less = True)
        x2 = self.gc2(x1,adj)
        return x1, x2               